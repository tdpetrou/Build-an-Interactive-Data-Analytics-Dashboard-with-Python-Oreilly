# Creates the interactive dashboard using the dash library
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative


# Instantiate the dash app
app = dash.Dash(
    __name__,
    # stylesheet for dash_bootstrap_components
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/flatly/bootstrap.min.css"
    ],
)

# URLs
HOME_URL = "https://www.dunderdata.com"
THIS_COURSE_URL = f"{HOME_URL}/build-an-interactive-data-analytics-dashboard-with-python"

# A blue and orange shade for actual and prediction from the Tableau 10 color sequence
COLORS = qualitative.T10[:2]

# Read in the data that powers the app
# SUMMARY table is much smaller, just one row per area and
# gets displayed as the data table on left hand side.
# ALL_DATA is all of the historical data and future predictions
# for all areas. It is used to make graphs in upper right hand side
SUMMARY = pd.read_csv("data/summary.csv", index_col="group", parse_dates=["date"])
ALL_DATA = pd.read_csv(
    "data/all_data.csv", index_col=["group", "area", "date"], parse_dates=["date"]
).sort_index()
LAST_DATE = SUMMARY["date"].iloc[0]
FIRST_PRED_DATE = LAST_DATE + pd.Timedelta("1D")


def create_table(group):
    """
    Create world and usa data table in the upper left corner of the dashboard.
    This uses the Dash DataTable package.

    Parameters
    ----------
    group : "world" or "usa"

    Returns
    -------
    dash data_table of provided group
    """
    used_columns = [
        "area",
        "Deaths",
        "Cases",
        "Deaths per Million",
        "Cases per Million",
    ]
    df = SUMMARY.loc[group, used_columns]
    first_col = "Country" if group == "world" else "State"
    df = df.rename(columns={"area": first_col})

    # Create list of dictionaries for dash table with info on each column
    columns = [{"name": first_col, "id": first_col}]
    for name in df.columns[1:]:
        # The first column is area.
        # All of the other columns are numeric.
        # We explicitly tell dash their type and format
        col_info = {
            "name": name,
            "id": name,
            "type": "numeric",
            "format": {"specifier": ","},
        }
        columns.append(col_info)

    # dash requires the data to be a list of dictionaries
    data = df.sort_values("Deaths", ascending=False).to_dict("records")
    return dash_table.DataTable(
        id=f"{group}-table",
        columns=columns,
        data=data,
        # active_cell is the top left hand cell by default
        # this is the area that will be graphed to the right
        active_cell={"row": 0, "column": 0},
        # Keep the column names on top as you scroll down
        fixed_rows={"headers": True},
        sort_action="native",
        # Necessary to read the correct data after sorting
        derived_virtual_data=data,
        # CSS can be placed in style.css, but dash examples online put table CSS here
        style_table={
            # dash bug - when fixed_rows provided
            # both minHeight and height are required to set height
            "min-height": "85vh",
            "height": "85vh",
            "overflow-y": "scroll",
            "border-radius": "0px 0px 10px 10px",
        },
        # CSS for ALL cells
        style_cell={
            "white-space": "normal",
            "height": "auto",
            "font-family": "verdana",
        },
        # CSS for only header cells
        style_header={
            "text-align": "center",
            "font-size": 14,
        },
        # CSS for only data cells
        style_data={
            "font-size": 12,
        },
        # Conditional CSS styling for first column.
        # Underline name and change cursor to indicate
        # that it is clickable
        style_data_conditional=[
            {
                "if": {"column_id": first_col},
                "width": "120px",
                "text-align": "left",
                "text-decoration": "underline",
                "cursor": "pointer",
            },
            # Style every other row with a light background color
            {"if": {"row_index": "odd"}, "background-color": "#fafbfb"},
        ],
    )


def create_graphs(group, area):
    """
    Creates three plotly figures for cumulative, daily, and weekly
    totals. There are two plots in each figure - one for deaths and
    the other for cases. Each figure is placed within a dcc.graph object.

    Parameters
    ----------
    group : "world" or "usa"

    area : str, name of country or state

    Returns
    -------
    list of plotly figures, one for cumulative, daily, and weekly
    """
    df = ALL_DATA.loc[(group, area)]

    # Store actual and predicted DataFrames in a dictionary
    df_dict = {"actual": df.loc[:LAST_DATE], "prediction": df.loc[FIRST_PRED_DATE:]}
    figs = create_figures(area)

    kinds = ["Deaths", "Cases"]
    daily_kinds = ["Daily Deaths", "Daily Cases"]

    # These next three functions update the figure in place
    make_cumulative_graphs(figs[0], df_dict, kinds)
    make_daily_graphs(figs[1], df_dict, daily_kinds)
    make_weekly_graphs(figs[2], df_dict, daily_kinds)
    return figs


def create_figures(title, n=3):
    """
    Called from create_graphs to create the three figures.
    make_subplots is used to create two plots for each figure.
    The title of the figure is the area name.

    Parameters
    ----------
    title : str, title of figure

    n : int, number of figures to create

    Returns
    -------
    list of three plotly figures without plots, just layout properties set
    """
    figs = []
    annot_props = {
        "x": 0.1,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "showarrow": False,
        "font": {"size": 18},
    }
    for _ in range(n):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
        fig.update_layout(
            title={"text": title, "x": 0.5, "y": 0.97, "font": {"size": 20}},
            annotations=[
                {"y": 0.95, "text": "<b>Deaths</b>"},
                {"y": 0.3, "text": "<b>Cases</b>"},
            ],
            margin={"t": 40, "l": 50, "r": 10, "b": 0},
            legend={
                "x": 0.5,
                "y": -0.05,
                "xanchor": "center",
                "orientation": "h",
                "font": {"size": 15},
            },
        )
        fig.update_traces(showlegend=False, row=2, col=1)
        fig.update_traces(hovertemplate="%{x} - %{y:,}")
        fig.update_annotations(annot_props)
        figs.append(fig)
    return figs


############# Upper right hand corner line graphs ##############

# There are three different tabs each with two visible plots.
# One plot for deaths and the other for cases.
# The same plots are produced for Country and State.
# The tabs have graphs for cumulative total, daily, and weekly
# All of these functions update the figure in place and return None


def make_cumulative_graphs(fig, df_dict, kinds):
    """
    Make cumulative total line graph for deaths and cases
    Two scatter plots (with connected lines) are plotted on each plot.
    One for the actual (blue) and one for the predicted value (orange)

    Parameters
    ----------
    fig : plotly figure

    df_dict : dict mapping "actual" and "prediction" to respective DataFrames

    kinds : list of two strings ["Deaths", "Cases"] or ["Daily Deaths"/"Daily Cases"]

    Returns
    -------
    None, updates figures in place with `add_scatter` method
    """
    for row, kind in enumerate(kinds, start=1):
        for (name, df), color in zip(df_dict.items(), COLORS):
            fig.add_scatter(
                x=df.index,
                y=df[kind],
                mode="lines+markers",
                line={"color": color},
                showlegend=row == 1,
                name=name,
                row=row,
                col=1,
            )


def make_daily_graphs(fig, df_dict, kinds):
    """
    Make daily bar graphs for actual and predicted values of deaths and cases
    """
    for row, kind in enumerate(kinds, start=1):
        for (name, df), color in zip(df_dict.items(), COLORS):
            fig.add_bar(
                x=df.index,
                y=df[kind],
                marker={"color": color},
                showlegend=row == 1,
                name=name,
                row=row,
                col=1,
            )


def make_weekly_graphs(fig, df_dict, kinds):
    """
    Make weekly total scatter plots (with connected lines)
    for actual and predicted deaths and cases
    """
    # Get offset alias for pandas resample method to group by week
    offset = "W-" + LAST_DATE.strftime("%a").upper()
    df_dict = {
        name: df.resample(offset, kind="timestamp", closed="right")[kinds].sum()
        for name, df in df_dict.items()
    }

    for row, kind in enumerate(kinds, start=1):
        for (name, df), color in zip(df_dict.items(), COLORS):
            fig.add_scatter(
                x=df.index,
                y=df[kind],
                mode="lines+markers",
                showlegend=row == 1,
                line={"color": color},
                name=name,
                row=row,
                col=1,
            )


########################## Map Creation  ##########################

# Two maps are created - one for the world, and the other for USA
# Each map can be colored by deaths, cases, deaths per million,
# or cases per million. Hover text displays all of the info


def hover_text(x):
    """
    Called from the DataFrame.apply method within the create_map function
    to create a few lines of text display when hovering over an area
    with info on deaths, cases, deaths per million, and cases per million.

    Parameters
    ----------
    x : row of a pandas DataFrame as a Series

    Returns
    -------
    A string using html tags for bold (<b>) and line breaks (<br>)
    """
    name = x["area"]
    deaths = x["Deaths"]
    cases = x["Cases"]
    deathsm = x["Deaths per Million"]
    casesm = x["Cases per Million"]
    return (
        f"<b>{name}</b><br>"
        f"Deaths - {deaths:,.0f}<br>"
        f"Cases - {cases:,.0f}<br>"
        f"Deaths per Million - {deathsm:,.0f}<br>"
        f"Cases per Million - {casesm:,.0f}<br>"
    )


def create_map(group, radio_value):
    """
    Creates a choropleth map based on the passed value of group, coloring it
    by the chosen radio_value (deaths, cases, deaths/MM, cases/MM)

    Parameters
    ----------
    group : "world" or "usa"

    radio_value : "Deaths", "Cases", "Deaths per Million" or "Cases per Million"

    Returns
    -------
    plotly figure with choropleth trace
    """
    # Color countries/states with at least 500k in population
    df = SUMMARY.loc[group].query("population > .5")
    lm = None if group == "world" else "USA-states"
    proj = "robinson" if group == "world" else "albers usa"
    text = df.apply(hover_text, axis=1)

    fig = go.Figure()
    fig.add_choropleth(
        locations=df["code"],
        z=df[radio_value],
        zmin=0,
        locationmode=lm,
        colorscale="orrd",
        marker_line_width=0.5,
        text=text,
        hoverinfo="text",
        colorbar=dict(len=0.6, x=1, y=0.5),
    )
    fig.update_layout(
        geo={
            # The range of the latitude and longitude is reduced
            # to remove the poles and some of the pacific ocean.
            # This only affects the world map and is ignored for usa
            "lataxis": {"range": [-50, 68]},
            "lonaxis": {"range": [-130, 150]},
            "projection": {"type": proj},
            "showframe": False,
        },
        margin={"t": 0, "l": 10, "r": 10, "b": 0},
    )
    return fig


############################### Create Layout ###############################

# Some of these components come from the dash_bootstrap_components library,
# aliased as `dbc`. It is a third party library that provides many
# nice bootstrap components such as navbars and cards specifically for dash


def create_navbar():
    """
    Creates the navigation bar at the top of the page

    Returns
    -------
    A navigation bar from dash_bootstrap_components
    """
    nav_items = dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Coronavirus Dashboard", href="/")),
            dbc.NavItem(
                dbc.NavLink(
                    "Learn How to Build this Dashboard",
                    href=THIS_COURSE_URL,
                    target="_blank",
                )
            ),
        ],
        navbar=True,
    )

    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [dbc.Col(html.Img(src="assets/dark_logo.png", height="30px"))],
                    align="center",
                ),
                href=HOME_URL,
                target="_blank",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(nav_items, id="navbar-collapse", navbar=True),
        ],
        color="primary",
        dark=True,
    )


def create_main_page():
    """
    Large function that creates all the components of the application
    under the navigation bar.

    Other functions are created within this function that are only
        called in create_main_page

    Returns
    -------
    A dash_html_components Div with multiple other Divs
    """
    # Create two pieces of text horizontally just under the navbar
    # the title of the application and the day it was last updated
    last_update_text = LAST_DATE.strftime("%B %d, %Y")
    top_info = html.Div(
        [
            html.H3("Coronavirus Forecasting Dashboard", id="info-title"),
            html.H4(f"Data updated through {last_update_text}", id="data-update"),
        ],
        className="top-info",
    )

    ######################## Left hand side Components ########################

    def create_tab(content, label, value):
        """
        Function to create a single tab

        Parameters
        ----------
        content : Any dash component, data_table or dcc.Graph in our case

        label : str, label visible to the user

        value : str, underlying value not visible to the user
        """
        return dcc.Tab(
            content,
            label=label,
            value=value,
            id=f"{value}-tab",
            className="single-tab",
            selected_className="single-tab--selected",
        )

    # Create the left hand side data tables using the function create_table
    world_table = create_table("world")
    usa_table = create_table("usa")

    # Create two individual tabs (world/usa) above the table.
    # They control which table, graphs, and map are active
    world_tab = create_tab(world_table, "World", "world")
    usa_tab = create_tab(usa_table, "US States", "usa")

    # Create the container for the individual tabs
    # Notice the slight difference in object name - Tab vs Tabs
    # This table_tabs object is the big container for the objects
    # in the left hand side of the dashboard
    table_tabs = dcc.Tabs(
        [world_tab, usa_tab],
        mobile_breakpoint=1000,
        className="tabs-container",
        id="table-tabs",
        value="world",  # default value
    )

    #################### Upper right hand side Components ####################

    # Each plotly figure must be contained in a dcc.Graph object
    # Here, we set configuration options for dcc.Graph object
    # Check https://dash.plotly.com/dash-core-components/graph for all options
    graph_kwargs = {
        "config": {"displayModeBar": False, "responsive": True},
        "className": "top-graphs",
    }
    cumulative_graph = dcc.Graph(id="cumulative-graph", **graph_kwargs)
    daily_graph = dcc.Graph(id="daily-graph", **graph_kwargs)
    weekly_graph = dcc.Graph(id="weekly-graph", **graph_kwargs)

    # Create three individual tabs for cumulative, daily, and weekly graphs
    cumulative_tab = create_tab(cumulative_graph, "Cumulative", "cumulative")
    daily_tab = create_tab(daily_graph, "Daily", "daily")
    weekly_tab = create_tab(weekly_graph, "Weekly", "weekly")

    # Container for the above three tabs. graph_tabs is the container for
    # all components in the upper right hand side of the application
    graph_tabs = dcc.Tabs(
        [cumulative_tab, daily_tab, weekly_tab],
        mobile_breakpoint=1000,
        className="tabs-container",
        id="graph-tabs",
        value="cumulative",  # default value
    )

    #################### Lower right hand side components ####################

    # Radio buttons for controlling map coloring.
    # dash has radio buttons, but dbc has the ability to
    # control the checked (selected) label.
    radio_items = dbc.RadioItems(
        options=[
            {"label": "Deaths", "value": "Deaths"},
            {"label": "Cases", "value": "Cases"},
            {"label": "Deaths per Million", "value": "Deaths per Million"},
            {"label": "Cases per Million", "value": "Cases per Million"},
        ],
        value="Deaths",  # default value
        id="map-radio-items",
        labelCheckedClassName="label-radio-checked",
        inputClassName="label-radio-input",
    )

    # The choropleth map, like all plotly figures, must be within a dcc.Graph object
    map_graph = dcc.Graph(
        id="map-graph", config={"displayModeBar": False, "responsive": True}
    )

    # major container for all components in the lower right hand side
    map_div = html.Div([radio_items, map_graph], id="map-div")

    ########################### Left side column ###########################

    # four bootstrap cards containing worldwide totals and predictions

    df_summary = SUMMARY.loc["world"]
    df_all = ALL_DATA.loc["world"]
    last_pred_date = ALL_DATA.iloc[-1].name[-1]
    last_pred_date_txt = last_pred_date.strftime("%b %d, %Y")
    ww_deaths, ww_cases = df_summary[["Deaths", "Cases"]].sum()
    ww_deathsp, ww_casesp = (
        df_all.droplevel(0).loc[last_pred_date, ["Deaths", "Cases"]].sum()
    )

    # Function to create individual bootstrap cards
    def create_card(header, number):
        return dbc.Card(
            [
                dbc.CardHeader(html.H5(header), className="side-card-header"),
                dbc.CardBody([html.H6(f"{number:,.0f}")]),
            ],
            color="primary",
            className="side-card",
            outline=True,
        )

    # A dictionary to hold the headers and number of each card
    header_numbers = {
        "Worldwide Deaths": ww_deaths,
        "Worldwide Cases": ww_cases,
        f"Worldwide Deaths Predicted by {last_pred_date_txt}": ww_deathsp,
        f"Worldwide Cases Predicted by {last_pred_date_txt}": ww_casesp,
    }

    # Create a list of 4 cards by iterating through dictionary
    cards = [create_card(header, number) for header, number in header_numbers.items()]

    # container for all cards on the side
    all_side_cards = html.Div(cards, id="all-side-cards")

    ###################### Containers for all components ######################

    # This div uses CSS Grid (display: grid) in assets/style.css to layout
    # the major components of the dashboard
    grid_container = html.Div(
        [all_side_cards, table_tabs, graph_tabs, map_div], id="grid-container"
    )
    # A container to place the grid below the top info
    container = html.Div([top_info, grid_container], id="container")
    return container


############################# Final Layout ##################################
# Place the navbar and container into a single div and set the final layout

navbar = create_navbar()
container = create_main_page()
app.layout = html.Div([navbar, container])


################################## Callbacks ##################################

# These functions are triggered by a user clicking on an object


@app.callback(
    [
        Output("cumulative-graph", "figure"),
        Output("daily-graph", "figure"),
        Output("weekly-graph", "figure"),
    ],
    [
        Input("world-table", "active_cell"),
        Input("usa-table", "active_cell"),
        Input("table-tabs", "value"),
    ],
    [
        State("world-table", "derived_virtual_data"),
        State("usa-table", "derived_virtual_data"),
    ],
)
def change_area_graphs(world_cell, usa_cell, group, world_data, usa_data):
    """
    Creates the three graphs in the upper right hand side

    Triggered when a country/state in the data table is clicked or when
    the World/USA States tab is selected.

    Passes in BOTH the world and usa data table active cells and their
    data regardless of which table is actually clicked. This is done because
    Dash only allows each component to appear in exactly one output. Therefore,
    we cannot make separate functions for world/usa data table clicks because
    the same outputs ("cumulative-graph", "daily-graph", and "weekly-graph")
    are used.

    We must use the tab value ("world" or "usa") to determine which table was
    selected by the user and which is the actual active cell.

    Inputs
    ------
    world_cell/usa_cell: Active cell for world/usa table. Dash passes this in
        as a dictionary with 'row', 'column', 'row_id', and 'column_id' keys.
        Unfortunately, the actual value (e.g. "Texas") is not provided. We
        must use the table data, "derived_virtual_data" to look up the value.

    group: String value of the current tab above the data table.
            Values are either "world" or "usa".

    world_data/usa_data: All data from the data table as a list of dictionaries
        One dictionary per row is given. The dictionary maps each column name
        to its value.

    Outputs
    -------
    Three plotly figures for cumulative, daily, and weekly graphs
    """
    area, cell, data = "Country", world_cell, world_data
    if group == "usa":
        area, cell, data = "State", usa_cell, usa_data

    if cell and cell["column"] == 0:
        country_state = data[cell["row"]][area]
        return create_graphs(group, country_state)
    else:
        # When sorting a column, there is no active cell, so
        # world_cell and usa_cell are pased as `None` and this branch
        # of the conditional is run. Dash expects us to return
        # 3 figures. This exception must be raised to alert dash
        # that we don't need to do anything.
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("map-graph", "figure"),
    [Input("table-tabs", "value"), Input("map-radio-items", "value")],
)
def change_map(group, radio_value):
    """
    Toggles the map to world/usa views and also controls the coloring

    Triggered whenever the USA/World tab is selected or when one of the
    radio buttons above the map is clicked.

    Inputs
    ------
    group: String value of the current tab above the data table.
            Values are either "world" or "usa".

    radio_value: One of four string values of the selected radio button
    """
    return create_map(group, radio_value)


# Necessary for toggling navbar collapse on small screens
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True)
