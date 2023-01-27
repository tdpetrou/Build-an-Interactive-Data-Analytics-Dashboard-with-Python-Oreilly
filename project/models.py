import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess

GROUPS = "world", "usa"
KINDS = "cases", "deaths"
MIN_OBS = 15  # Minimum observations needed to make prediction


def general_logistic_shift(x, L, x0, k, v, s):
    return (L - s) / ((1 + np.exp(-k * (x - x0))) ** (1 / v)) + s


def optimize_func(params, x, y, model):
    y_pred = model(x, *params)
    error = y - y_pred
    return error


class CasesModel:
    def __init__(self, model, data, last_date, n_train, n_smooth, 
                 n_pred, L_n_min, L_n_max, **kwargs):
        """
        Smooths, trains, and predicts cases for all areas
        
        Parameters
        ----------
        model : function such as general_logistic_shift
        
        data : dictionary of data from all areas - result of PrepareData().run()
        
        last_date : str, last date to be used for training
        
        n_train : int, number of preceding days to use for training
        
        n_smooth : integer, number of points used in LOWESS
        
        n_pred : int, days of predictions to make
        
        L_n_min, L_n_max : int, min/max number of days used to estimate L_min/L_max
        
        **kwargs : extra keyword arguments passed to scipy's least_squares function
        """
        # Set basic attributes
        self.model = model
        self.data = data
        self.last_date = self.get_last_date(last_date)
        self.n_train = n_train
        self.n_smooth = n_smooth
        self.n_pred = n_pred
        self.L_n_min = L_n_min
        self.L_n_max = L_n_max
        self.kwargs = kwargs
        
        # Set attributes for prediction
        self.first_pred_date = pd.Timestamp(self.last_date) + pd.Timedelta("1D")
        self.pred_index = pd.date_range(start=self.first_pred_date, periods=n_pred)
        
    def get_last_date(self, last_date):
        # Use the most current date as the last actual date if not provided
        if last_date is None:
            return self.data['world_cases'].index[-1]
        else:
            return pd.Timestamp(last_date)
        
    def init_dictionaries(self):
        # Create dictionaries to store results for each area
        # Executed first in `run` method
        self.smoothed = {'world_cases': {}, 'usa_cases': {}}
        self.bounds = {'world_cases': {}, 'usa_cases': {}}
        self.p0 = {'world_cases': {}, 'usa_cases': {}}
        self.params = {'world_cases': {}, 'usa_cases': {}}
        self.pred_daily = {'world_cases': {}, 'usa_cases': {}}
        self.pred_cumulative = {'world_cases': {}, 'usa_cases': {}}
        
        # Dictionary to hold DataFrame of actual and predicted values
        self.combined_daily = {}
        self.combined_cumulative = {}
        
        # Same as above, but stores smoothed and predicted values
        self.combined_daily_s = {}
        self.combined_cumulative_s = {}
        
    def smooth(self, s):
        s = s[:self.last_date]
        if s.values[0] == 0:
            # Filter the data if the first value is 0
            last_zero_date = s[s == 0].index[-1]
            s = s.loc[last_zero_date:]
            s_daily = s.diff().dropna()
        else:
            # If first value not 0, use it to fill in the 
            # first missing value
            s_daily = s.diff().fillna(s.iloc[0])

        # Don't smooth data with less than MIN_OBS values
        if len(s_daily) < MIN_OBS:
            return s_daily.cumsum()

        y = s_daily.values
        frac = self.n_smooth / len(y)
        x = np.arange(len(y))
        y_pred = lowess(y, x, frac=frac, is_sorted=True, return_sorted=False)
        s_pred = pd.Series(y_pred, index=s_daily.index).clip(0)
        s_pred_cumulative = s_pred.cumsum()
        
        if s_pred_cumulative[-1]  == 0:
            # Don't use smoothed values if they are all 0
            return s_daily.cumsum()
        
        last_actual = s.values[-1]
        last_smoothed = s_pred_cumulative.values[-1]
        s_pred_cumulative *= last_actual / last_smoothed
        return s_pred_cumulative
    
    def get_train(self, smoothed):
        # Filter the data for the most recent to capture new waves
        return smoothed.iloc[-self.n_train:]
    
    def get_L_limits(self, s):
        last_val = s[-1]
        last_pct = s.pct_change()[-1] + 1
        L_min = last_val * last_pct ** self.L_n_min
        L_max = last_val * last_pct ** self.L_n_max + 1
        L0 = (L_max - L_min) / 2 + L_min
        return L_min, L_max, L0
    
    def get_bounds_p0(self, s):
        L_min, L_max, L0 = self.get_L_limits(s)
        x0_min, x0_max = -50, 50
        k_min, k_max = 0.01, 0.5
        v_min, v_max = 0.01, 2
        s_min, s_max = 0, s[-1] + 0.01
        s0 = s_max / 2
        lower = L_min, x0_min, k_min, v_min, s_min
        upper = L_max, x0_max, k_max, v_max, s_max
        bounds = lower, upper
        p0 = L0, 0, 0.1, 0.1, s0
        return bounds, p0
    
    def train_model(self, s, bounds, p0):
        y = s.values
        n_train = len(y)
        x = np.arange(n_train)
        res = least_squares(optimize_func, p0, args=(x, y, self.model), bounds=bounds, **self.kwargs)
        return res.x
    
    def get_pred_daily(self, n_train, params):
        x_pred = np.arange(n_train - 1, n_train + self.n_pred)
        y_pred = self.model(x_pred, *params)
        y_pred_daily = np.diff(y_pred)
        return pd.Series(y_pred_daily, index=self.pred_index)
    
    def get_pred_cumulative(self, s, pred_daily):
        last_actual_value = s.loc[self.last_date]
        return pred_daily.cumsum() + last_actual_value
    
    def convert_to_df(self, gk):
        # convert dictionary of areas mapped to Series to DataFrames
        self.smoothed[gk] = pd.DataFrame(self.smoothed[gk]).fillna(0).astype('int')
        self.bounds[gk] = pd.concat(self.bounds[gk].values(), 
                                    keys=self.bounds[gk].keys()).T
        self.bounds[gk].loc['L'] = self.bounds[gk].loc['L'].round()
        self.p0[gk] = pd.DataFrame(self.p0[gk], index=['L', 'x0', 'k', 'v', 's'])
        self.p0[gk].loc['L'] = self.p0[gk].loc['L'].round()
        self.params[gk] = pd.DataFrame(self.params[gk], index=['L', 'x0', 'k', 'v', 's'])
        self.pred_daily[gk] = pd.DataFrame(self.pred_daily[gk])
        self.pred_cumulative[gk] = pd.DataFrame(self.pred_cumulative[gk])
        
    def combine_actual_with_pred(self):
        for gk, df_pred in self.pred_cumulative.items():
            df_actual = self.data[gk][:self.last_date]
            df_comb = pd.concat((df_actual, df_pred))
            self.combined_cumulative[gk] = df_comb
            self.combined_daily[gk] = df_comb.diff().fillna(df_comb.iloc[0]).astype('int')
            
            df_comb_smooth = pd.concat((self.smoothed[gk], df_pred))
            self.combined_cumulative_s[gk] = df_comb_smooth
            self.combined_daily_s[gk] = df_comb_smooth.diff().fillna(df_comb.iloc[0]).astype('int')

    def run(self):
        self.init_dictionaries()
        for group in GROUPS:
            gk = f'{group}_cases'
            df_cases = self.data[gk]
            for area, s in df_cases.items():
                smoothed = self.smooth(s)
                train = self.get_train(smoothed)
                n_train = len(train)
                if n_train < MIN_OBS:
                    bounds = np.full((2, 5), np.nan)
                    p0 = np.full(5, np.nan)
                    params = np.full(5, np.nan)
                    pred_daily = pd.Series(np.zeros(self.n_pred), index=self.pred_index)
                else:
                    bounds, p0 = self.get_bounds_p0(train)
                    params = self.train_model(train, bounds=bounds,  p0=p0)
                    pred_daily = self.get_pred_daily(n_train, params).round(0)
                pred_cumulative = self.get_pred_cumulative(s, pred_daily)
                
                # save results to dictionaries mapping each area to its result
                self.smoothed[gk][area] = smoothed
                self.bounds[gk][area] = pd.DataFrame(bounds, index=['lower', 'upper'], 
                                                     columns=['L', 'x0', 'k', 'v', 's'])
                self.p0[gk][area] = p0
                self.params[gk][area] = params
                self.pred_daily[gk][area] = pred_daily.astype('int')
                self.pred_cumulative[gk][area] = pred_cumulative.astype('int')
            self.convert_to_df(gk)
        self.combine_actual_with_pred()
        
    def plot_prediction(self, group, area, **kwargs):
        group_kind = f'{group}_cases'
        actual = self.data[group_kind][area]
        pred = self.pred_cumulative[group_kind][area]
        first_date = self.last_date - pd.Timedelta(self.n_train, 'D')
        last_pred_date = self.last_date + pd.Timedelta(self.n_pred, 'D')
        actual.loc[first_date:last_pred_date].plot(label='Actual', **kwargs)
        pred.plot(label='Predicted').legend()


class DeathsModel:
    def __init__(self, data, last_date, cm, lag, period):
        """
        Build simple model based on CFR to predict deaths for all areas

        Parameters
        ----------
        data : dictionary of data from all areas - result of PrepareData().run()

        last_date : str, last date to be used for training

        cm : CasesModel instance after calling `run` method
        
        lag : int, number of days between cases and deaths, used to calculate CFR
        
        period : int, window size of number of days to calculate CFR
        """
        self.data = data
        self.last_date = self.get_last_date(last_date)
        self.cm = cm
        self.lag = lag
        self.period = period
        self.pred_daily = {}
        self.pred_cumulative = {}
        
        # Dictionary to hold DataFrame of actual and predicted values
        self.combined_daily = {}
        self.combined_cumulative = {}
        
    def get_last_date(self, last_date):
        if last_date is None:
            return self.data['world_cases'].index[-1]
        else:
            return pd.Timestamp(last_date)
        
    def calculate_cfr(self):
        first_day_deaths = self.last_date - pd.Timedelta(f'{self.period}D')
        last_day_cases = self.last_date - pd.Timedelta(f'{self.lag}D')
        first_day_cases = last_day_cases - pd.Timedelta(f'{self.period}D')

        cfr = {}
        for group in GROUPS:
            deaths = self.data[f'{group}_deaths']
            cases = self.data[f'{group}_cases']
            deaths_total = deaths.loc[self.last_date] - deaths.loc[first_day_deaths]
            cases_total = cases.loc[last_day_cases] - cases.loc[first_day_cases]
            cfr[group] = (deaths_total / cases_total).fillna(0.01)
        return cfr
    
    def run(self):
        self.cfr = self.calculate_cfr()
        for group in GROUPS:
            group_cases = f'{group}_cases'
            group_deaths = f'{group}_deaths'
            cfr_start_date = self.last_date - pd.Timedelta(f'{self.lag}D')
            
            daily_cases_smoothed = self.cm.combined_daily_s[group_cases]
            pred_daily = daily_cases_smoothed[cfr_start_date:] * self.cfr[group]
            pred_daily = pred_daily.iloc[:self.cm.n_pred]
            pred_daily.index = self.cm.pred_daily[group_cases].index
            
            # Use repeated rolling average to smooth out the predicted deaths
            for i in range(5):
                pred_daily = pred_daily.rolling(14, min_periods=1, center=True).mean()
            
            pred_daily = pred_daily.round(0).fillna(0).astype("int")
            self.pred_daily[group_deaths] = pred_daily
            last_deaths = self.data[group_deaths].loc[self.last_date]
            self.pred_cumulative[group_deaths] = pred_daily.cumsum() + last_deaths
        self.combine_actual_with_pred()
            
    def combine_actual_with_pred(self):
        for gk, df_pred in self.pred_cumulative.items():
            df_actual = self.data[gk][:self.last_date]
            df_comb = pd.concat((df_actual, df_pred))
            self.combined_cumulative[gk] = df_comb
            self.combined_daily[gk] = df_comb.diff().fillna(df_comb.iloc[0]).astype('int')
            
    def plot_prediction(self, group, area, **kwargs):
        group_kind = f'{group}_deaths'
        actual = self.data[group_kind][area]
        pred = self.pred_cumulative[group_kind][area]
        first_date = self.last_date - pd.Timedelta(60, 'D')
        last_pred_date = self.last_date + pd.Timedelta(30, 'D')
        actual.loc[first_date:last_pred_date].plot(label='Actual', **kwargs)
        pred.plot(label='Predicted').legend()
