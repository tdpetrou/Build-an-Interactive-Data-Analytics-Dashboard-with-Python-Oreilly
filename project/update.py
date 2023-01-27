# Script to run from the command line to update the data.
# Use without options to get data up to the current date
# python update.py
#
# Use with a single option, an 8-character date to get data for
# one particular date
# python update.py 20200720

import sys
from prepare import PrepareData, combine_all_data, create_summary_table
from models import CasesModel, DeathsModel, general_logistic_shift

# Constants for CasesModel - Feel free to change these
N_TRAIN = 60   # Number of observations used in training
N_SMOOTH = 15  # Number of observations used in smoothing
N_PRED = 56    # Number of new observations to predict
L_N_MIN = 5    # Number of days of exponential growth for L min boundary
L_N_MAX = 50   # Number of days of exponential growth for L max boundary

# Constants for DeathsModel - Feel free to change these
LAG = 15     # Number of days to lag cases for calculation of CFR
PERIOD = 30  # Number of days to total for CFR

if __name__ == "__main__":
    if len(sys.argv) == 1:
        last_date = None
    elif len(sys.argv) == 2:
        last_date = sys.argv[1]
    else:
        raise TypeError(
            """
            When calling `python update.py` from the command line,
            pass 0 or 1 arguments.
                0 arguments: make prediction for latest data (downloads latest data)
                1 argument: provide the last date that the model will see, i.e. 20200720
            """
        )
    data = PrepareData().run()
    cm = CasesModel(
        model=general_logistic_shift,
        data=data,
        last_date=last_date,
        n_train=N_TRAIN,
        n_smooth=N_SMOOTH,
        n_pred=N_PRED,
        L_n_min=L_N_MIN,
        L_n_max=L_N_MAX,
    )
    cm.run()

    dm = DeathsModel(data=data, last_date=last_date, cm=cm, lag=15, period=30)
    dm.run()

    df_all = combine_all_data(cm, dm)
    create_summary_table(df_all, cm.last_date)
