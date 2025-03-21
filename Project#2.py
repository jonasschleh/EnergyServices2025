import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# EXTERNAL STYLESHEETS
# -------------------------------
# Use Bootstrap (Bootswatch Flatly) and Google Fonts (Montserrat) for a modern look.
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css',
    'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap'
]

# -------------------------------
# DATA LOADING AND PREPROCESSING
# -------------------------------

def load_north_tower_data():
    # Load 2017 and 2018 power consumption data and parse dates.
    df_2017 = pd.read_csv("IST_North_Tower_2017.csv")
    df_2018 = pd.read_csv("IST_North_Tower_2018.csv")
    df_2017["Date_start"] = pd.to_datetime(df_2017["Date_start"], dayfirst=True)
    df_2018["Date_start"] = pd.to_datetime(df_2018["Date_start"], dayfirst=True)
    # Concatenate the dataframes into one.
    return pd.concat([df_2017, df_2018], ignore_index=True)

def load_meteo_data():
    # Load meteo data (2017-2018) and convert date/time information.
    df_meteo = pd.read_csv("IST_meteo_data_2017_2018_2019.csv")
    df_meteo["yyyy-mm-dd hh:mm:ss"] = pd.to_datetime(df_meteo["yyyy-mm-dd hh:mm:ss"])
    # Exclude data from 2019.
    df_meteo = df_meteo[df_meteo["yyyy-mm-dd hh:mm:ss"].dt.year < 2019]
    # Floor the datetime to hourly values.
    df_meteo["hour"] = df_meteo["yyyy-mm-dd hh:mm:ss"].dt.floor("h")
    # Aggregate the meteo data hourly.
    df_meteo_agg = df_meteo.groupby("hour").agg({
        "temp_C": "mean", "HR": "mean", "windSpeed_m/s": "mean", "windGust_m/s": "mean",
        "pres_mbar": "mean", "solarRad_W/m2": "mean", "rain_mm/h": "sum", "rain_day": "mean"
    }).reset_index()
    # Rename the column for merging.
    df_meteo_agg.rename(columns={"hour": "Date_start"}, inplace=True)
    df_meteo_agg["Date_start"] = pd.to_datetime(df_meteo_agg["Date_start"])
    return df_meteo_agg

def get_final_data():
    # Merge North Tower and meteo data for 2017-2018.
    df_nt = load_north_tower_data()
    df_meteo = load_meteo_data()
    df_final = pd.merge(df_nt, df_meteo, on="Date_start", how="left")

    # -------------------------------
    # DATA CLEANING
    # -------------------------------
    # Remove power values below threshold and fill missing values.
    df_final.loc[df_final["Power_kW"] < 10, "Power_kW"] = np.nan
    df_final["Power_kW"] = df_final["Power_kW"].ffill()
    df_final.ffill(inplace=True)

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    # Create lag features for power consumption.
    df_final["Power_kW-1"] = df_final["Power_kW"].shift(1)
    df_final["Power_kW-3"] = df_final["Power_kW"].shift(3)
    df_final["Power_kW-6"] = df_final["Power_kW"].shift(6)
    df_final["Power_kW-24"] = df_final["Power_kW"].shift(24)

    # Daily Heating Degree Days (HDD) and Cooling Degree Days (CDD)
    def calculate_daily_hdd(df, base_temp=16):
        daily_avg_temp = df.groupby(df["Date_start"].dt.date)["temp_C"].mean()
        daily_hdd = (base_temp - daily_avg_temp).clip(lower=0)
        return df["Date_start"].dt.date.map(daily_hdd)
    def calculate_daily_cdd(df, base_temp=21):
        daily_avg_temp = df.groupby(df["Date_start"].dt.date)["temp_C"].mean()
        daily_cdd = (daily_avg_temp - base_temp).clip(lower=0)
        return df["Date_start"].dt.date.map(daily_cdd)
    df_final["HDD_Daily"] = calculate_daily_hdd(df_final)
    df_final["CDD_Daily"] = calculate_daily_cdd(df_final)

    # Add cyclic features for hour of the day.
    df_final["Sin_Hour"] = np.sin(2 * np.pi * df_final["Date_start"].dt.hour / 24)
    df_final["Cos_Hour"] = np.cos(2 * np.pi * df_final["Date_start"].dt.hour / 24)

    # Time-related features: weekday, weekend, holiday, night, academic and Christmas breaks.
    df_final["Weekday"] = df_final["Date_start"].dt.weekday
    df_final["Weekend"] = df_final["Weekday"].apply(lambda x: 1 if x in [5, 6] else 0)
    prt_holidays = holidays.country_holidays('PT', years=[2017, 2018])
    def is_holiday(date):
        return 1 if date in prt_holidays else 0
    df_final["Holiday"] = df_final["Date_start"].dt.date.apply(is_holiday)
    df_final["Night"] = df_final["Date_start"].dt.hour.apply(lambda x: 0 if 6 <= x < 21 else 1)
    df_final["Academic_Break"] = df_final["Date_start"].apply(
        lambda dt: 1 if ((dt >= pd.to_datetime("2017-07-31") and dt <= pd.to_datetime("2017-09-10"))
                         or (dt >= pd.to_datetime("2018-07-31") and dt <= pd.to_datetime("2018-09-13")))
        else 0
    )
    df_final["Christmas_Break"] = df_final["Date_start"].apply(
        lambda dt: 1 if ((dt >= pd.to_datetime("2017-12-23") and dt <= pd.to_datetime("2017-12-29"))
                         or (dt >= pd.to_datetime("2018-12-22") and dt <= pd.to_datetime("2018-12-31")))
        else 0
    )
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    df_final.dropna(inplace=True)
    return df_final

def load_2019_data():
    # Load and process the 2019 test data.
    df_2019 = pd.read_csv("testData_2019_NorthTower.csv")
    df_2019.rename(columns={"Date": "Date_start", "North Tower (kWh)": "Power_kW"}, inplace=True)
    df_2019["Date_start"] = pd.to_datetime(df_2019["Date_start"])
    df_2019["Power_kW"] = df_2019["Power_kW"].ffill()
    df_2019.ffill(inplace=True)

    # Create lag features.
    df_2019["Power_kW-1"] = df_2019["Power_kW"].shift(1)
    df_2019["Power_kW-3"] = df_2019["Power_kW"].shift(3)
    df_2019["Power_kW-6"] = df_2019["Power_kW"].shift(6)
    df_2019["Power_kW-24"] = df_2019["Power_kW"].shift(24)

    # If available, compute temperature-related features.
    if "temp_C" in df_2019.columns:
        def calculate_daily_hdd(df, base_temp=16):
            daily_avg_temp = df.groupby(df["Date_start"].dt.date)["temp_C"].mean()
            daily_hdd = (base_temp - daily_avg_temp).clip(lower=0)
            return df["Date_start"].dt.date.map(daily_hdd)
        def calculate_daily_cdd(df, base_temp=21):
            daily_avg_temp = df.groupby(df["Date_start"].dt.date)["temp_C"].mean()
            daily_cdd = (daily_avg_temp - base_temp).clip(lower=0)
            return df["Date_start"].dt.date.map(daily_cdd)
        df_2019["HDD_Daily"] = calculate_daily_hdd(df_2019)
        df_2019["CDD_Daily"] = calculate_daily_cdd(df_2019)
        df_2019["Sin_Hour"] = np.sin(2 * np.pi * df_2019["Date_start"].dt.hour / 24)
        df_2019["Cos_Hour"] = np.cos(2 * np.pi * df_2019["Date_start"].dt.hour / 24)

    # Time features for 2019.
    df_2019["Weekday"] = df_2019["Date_start"].dt.weekday
    df_2019["Weekend"] = df_2019["Weekday"].apply(lambda x: 1 if x in [5, 6] else 0)
    prt_holidays = holidays.country_holidays('PT', years=[2019])
    def is_holiday(date):
        return 1 if date in prt_holidays else 0
    df_2019["Holiday"] = df_2019["Date_start"].dt.date.apply(is_holiday)
    df_2019["Night"] = df_2019["Date_start"].dt.hour.apply(lambda x: 0 if 6 <= x < 21 else 1)
    df_2019["Academic_Break"] = df_2019["Date_start"].apply(
        lambda dt: 1 if (dt >= pd.to_datetime("2019-07-31") and dt <= pd.to_datetime("2019-09-13"))
                       else 0
    )
    df_2019["Christmas_Break"] = df_2019["Date_start"].apply(
        lambda dt: 1 if (dt >= pd.to_datetime("2019-12-22") and dt <= pd.to_datetime("2019-12-31"))
                       else 0
    )
    df_2019.ffill(inplace=True)
    df_2019.bfill(inplace=True)
    df_2019.dropna(inplace=True)
    return df_2019

# Load the datasets.
df_final = get_final_data()   # 2017-2018 data
df_2019 = load_2019_data()    # 2019 test data

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_month_bounds(year, month):
    """Return start and end timestamps for the specified year and month."""
    year = int(year)
    month = int(month)
    start = pd.to_datetime(f"{year}-{month}-01")
    end = start + pd.offsets.MonthEnd(1)
    return start, end

def filter_data_by_range(df, start_perc, end_perc):
    """Return a slice of the dataframe based on percentage indices."""
    n = len(df)
    start_idx = int(n * start_perc / 100)
    end_idx = int(n * end_perc / 100)
    return df.iloc[start_idx:end_idx]

# -------------------------------
# DASHBOARD SETUP
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Power Forecast Dashboard"
container_style = {"margin": "20px", "fontFamily": "'Montserrat', sans-serif"}

####
server=app.server
####

# Define the layout for the dashboard.
app.layout = html.Div(style=container_style, children=[
    # -------------------------------
    # PINNED CONTAINER (Header & Controls)
    # -------------------------------
    html.Div([
        # Header Row with title and logo.
        html.Div([
            html.Div([
                html.H1("IST North Tower Power Forecast Dashboard",
                        style={"margin": "0", "padding": "10px", "fontWeight": "bold"})
            ], className="col-md-10"),
            html.Div([
                html.Img(src="https://diaaberto.tecnico.ulisboa.pt/files/sites/178/ist_a_rgb_pos-1.png",
                         style={"height": "100px", "margin": "10px"})
            ], className="col-md-2")
        ],
        className="row",
        style={
            "backgroundColor": "#CCEBF9",  # Blue header background.
            "padding": "10px",
            "alignItems": "center"
        }),
        # Control Row for Display Mode, Year, and Model selections.
        html.Div([
            html.Div([
                html.Label("Display Mode:", style={"fontWeight": "bold"}),
                dcc.RadioItems(
                    id="display-mode",
                    options=[
                        {"label": "Raw Data", "value": "raw"},
                        {"label": "Comparison", "value": "comparison"},
                        {"label": "Model Info (Metrics)", "value": "model_info"}
                    ],
                    value="raw",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                )
            ], className="col-md-4", style={"padding": "10px"}),
            html.Div([
                html.Label("Select Year:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[
                        {"label": "2017", "value": "2017"},
                        {"label": "2018", "value": "2018"},
                        {"label": "2019", "value": "2019"}
                    ],
                    value="2018",
                    clearable=False,
                    style={"width": "400px"}
                )
            ], className="col-md-4", style={"padding": "10px"}),
            html.Div([
                html.Label("Select Model:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "Random Forest", "value": "Random Forest"},
                        {"label": "Linear Regression", "value": "Linear Regression"},
                        {"label": "XGBoost", "value": "XGBoost"}
                    ],
                    value="rf",
                    clearable=False,
                    style={"width": "400px"}
                )
            ], className="col-md-4", style={"padding": "10px"}),
        ],
        className="row",
        style={
            "backgroundColor": "#ffffff",   # White background for controls.
            "borderBottom": "1px solid #46555f",
            "paddingBottom": "10px"
        })
    ],
    style={
        "position": "fixed",  # Keep header pinned to the top.
        "top": 0,
        "width": "100%",
        "zIndex": 1000,
    }),

    # -------------------------------
    # MAIN CONTENT AREA
    # -------------------------------
    html.Div([
        # Auxiliary Controls: Date Picker, Chart Type, and Additional Features.
        html.Div(id="aux-controls", children=[
            html.Div([
                html.Label("Select Date Range:", style={"fontWeight": "bold"}),
                html.Div([
                    dcc.DatePickerRange(
                        id="date-picker-range",
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                        min_date_allowed=pd.to_datetime("2017-01-01"),
                        max_date_allowed=pd.to_datetime("2019-12-31")
                    ),
                    html.Button("Clear Dates", id="clear-dates-button", n_clicks=0)
                ], style={"padding": "10px"})
            ], className="col-md-6", style={"padding": "10px"}),
            # Chart Type control (visible only in Raw Data mode).
            html.Div([
                html.Label("Select Chart Type:", style={"fontWeight": "bold"}),
                dcc.RadioItems(
                    id="chart-type",
                    options=[
                        {"label": "Line Chart", "value": "line"},
                        {"label": "Bar Chart", "value": "bar"},
                        {"label": "Scatter Plot", "value": "scatter"}
                    ],
                    value="line",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                )
            ], id="chart-type-container", className="col-md-4", style={"padding": "10px"}),
            html.Div([
                html.Label("Additional Feature:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[
                        {"label": "solarRad_W/m2", "value": "solarRad_W/m2"},
                        {"label": "Weekday", "value": "Weekday"},
                        {"label": "Weekend", "value": "Weekend"},
                        {"label": "HDD", "value": "HDD"},
                        {"label": "CDD", "value": "CDD"},
                        {"label": "Power_kW-1", "value": "Power_kW-1"},
                        {"label": "Power_kW-3", "value": "Power_kW-3"},
                        {"label": "Power_kW-6", "value": "Power_kW-6"},
                        {"label": "Power_kW-24", "value": "Power_kW-24"},
                        {"label": "HDD_Daily", "value": "HDD_Daily"},
                        {"label": "CDD_Daily", "value": "CDD_Daily"},
                        {"label": "Sin_Hour", "value": "Sin_Hour"},
                        {"label": "Cos_Hour", "value": "Cos_Hour"},
                        {"label": "Holiday", "value": "Holiday"},
                        {"label": "Night", "value": "Night"},
                        {"label": "Academic_Break", "value": "Academic_Break"},
                        {"label": "Christmas_Break", "value": "Christmas_Break"}
                    ],
                    value=[],  # Allow multiple selections.
                    multi=True,
                    style={"width": "100%"}
                )
            ], className="col-md-6", style={"padding": "10px"})
        ], className="row"),
        # Graph Display.
        dcc.Graph(id="main-graph"),
        # Slider: Adjust data range (hidden when in model_info mode).
        html.Div(id="slider-container", children=[
            html.Label("Adjust Plot Range (percentage):", style={"fontWeight": "bold", "marginTop": "20px"}),
            dcc.RangeSlider(
                id="range-slider",
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks={0: "0%", 50: "50%", 100: "100%"}
            )
        ], style={"padding": "10px", "marginTop": "20px"})
    ],
    style={
        "paddingTop": "220px"  # Ensure main content is pushed below the pinned header.
    }),

    # -------------------------------
    # FOOTER SECTION
    # -------------------------------
    html.Div([
        html.P([
            "Dashboard made by Jonas Schleh (ist1115359) ",
            html.A(
                html.Img(
                    src="https://images.rawpixel.com/image_png_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjk4Mi1kNS0xMF8xLnBuZw.png",
                    style={"height": "20px", "vertical-align": "middle", "margin-left": "5px"}
                ),
                href="https://www.linkedin.com/in/jonas-schleh-931bb4234/",
                target="_blank"
            )
        ],
        style={"margin": "0", "textAlign": "center", "fontWeight": "bold", "fontSize": "12px"})
    ],
    style={
        "backgroundColor": "#CCEBF9",  # Blue background matching the header.
        "padding": "10px",
        "marginTop": "20px",
        "width": "100%"
    })
])

# -------------------------------
# CALLBACKS
# -------------------------------

# Callback: Toggle visibility of auxiliary controls based on display mode.
@app.callback(
    Output("aux-controls", "style"),
    [Input("display-mode", "value")]
)
def toggle_aux_controls(display_mode):
    # Show auxiliary controls only for "raw" and "comparison" modes.
    if display_mode in ["raw", "comparison"]:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Callback: Clear date picker values when "Clear Dates" button is clicked.
@app.callback(
    [Output("date-picker-range", "start_date"),
     Output("date-picker-range", "end_date")],
    [Input("clear-dates-button", "n_clicks")]
)
def clear_dates(n_clicks):
    # If the button is clicked, reset the date picker values.
    if n_clicks and n_clicks > 0:
        return None, None
    return dash.no_update, dash.no_update

# Callback: Hide the slider container when display mode is "model_info".
@app.callback(
    Output("slider-container", "style"),
    [Input("display-mode", "value")]
)
def toggle_slider(display_mode):
    if display_mode == "model_info":
        return {"display": "none"}
    else:
        return {"display": "block", "padding": "10px", "marginTop": "20px"}

# Helper: Filter the dataframe based on a percentage range.
def filter_data_by_range(df, start_perc, end_perc):
    n = len(df)
    start_idx = int(n * start_perc / 100)
    end_idx = int(n * end_perc / 100)
    return df.iloc[start_idx:end_idx]

# Callback: Toggle the visibility of the chart type control.
@app.callback(
    Output("chart-type-container", "style"),
    [Input("display-mode", "value")]
)
def toggle_chart_type(display_mode):
    # Show chart type options only when in raw data mode.
    if display_mode == "raw":
        return {"display": "block", "padding": "10px"}
    else:
        return {"display": "none"}

# Main Callback: Update the graph based on all user inputs.
@app.callback(
    Output("main-graph", "figure"),
    [Input("display-mode", "value"),
     Input("year-dropdown", "value"),
     Input("model-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("feature-dropdown", "value"),
     Input("range-slider", "value"),
     Input("chart-type", "value")]
)
def update_graph(display_mode, year_choice, model_choice, start_date, end_date, feature_choice, slider_range, chart_type):
    # -------------------------------
    # DATA SELECTION AND FILTERING
    # -------------------------------
    # Select the appropriate dataset based on the chosen year.
    if year_choice in ["2017", "2018"]:
        df_year = df_final[df_final["Date_start"].dt.year == int(year_choice)]
    elif year_choice == "2019":
        df_year = df_2019
    else:
        df_year = df_final

    # Filter the data either by a date range or by a slider (percentage-based).
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_filtered = df_year[(df_year["Date_start"] >= start_date) & (df_year["Date_start"] <= end_date)]
    else:
        df_filtered = filter_data_by_range(df_year, slider_range[0], slider_range[1])

    # -------------------------------
    # DISPLAY MODE: MODEL INFO
    # -------------------------------
    if display_mode == "model_info":
        # If no valid model is selected, display an annotation message.
        if not model_choice or model_choice not in ["Random Forest", "Linear Regression", "XGBoost"]:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select a regression model.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="red", size=16)
            )
            return fig

        try:
            # Use 2019 data for model evaluation.
            X_df = df_2019.copy()
            features = ["Power_kW-1", "Power_kW-24", "solarRad_W/m2", "temp_C", "Weekday"]
            features = [feat for feat in features if feat in X_df.columns]
            X = X_df[features]
            y_true = X_df["Power_kW"]

            # Determine which model file to use based on user selection.
            if model_choice == "Random Forest":
                model_file = "random_forest_model_jonas.pkl"
            elif model_choice == "XGBoost":
                model_file = "xgboost_model_jonas.pkl"
            elif model_choice == "Linear Regression":
                model_file = "linear_regression_model_jonas.pkl"
            else:
                model_file = "random_forest_model_jonas.pkl"

            with open(model_file, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(X)

            # Compute evaluation metrics.
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Create a table to display metrics.
            table = go.Table(
                header=dict(values=["Metric", "Value"],
                            fill_color="rgba(0,157,224,0.2)",
                            align="left"),
                cells=dict(values=[["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "R²"],
                                   [f"{mae:.2f}", f"{mse:.2f}", f"{r2:.2f}"]],
                           fill_color="white",
                           align="left")
            )
            fig = go.Figure(data=[table])
            fig.update_layout(title=f"Model Evaluation Metrics for {model_choice.upper()}")
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error loading model info: {str(e)}",
                               xref="paper", yref="paper", showarrow=False)
            return fig

    # -------------------------------
    # DISPLAY MODE: RAW DATA
    # -------------------------------
    elif display_mode == "raw":
        fig = go.Figure()
        # Plot the main trace (Actual Power) based on the selected chart type.
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=df_filtered["Date_start"],
                y=df_filtered["Power_kW"],
                mode="lines",
                name="Actual Power",
                line={"width": 2}
            ))
        elif chart_type == "bar":
            fig.add_trace(go.Bar(
                x=df_filtered["Date_start"],
                y=df_filtered["Power_kW"],
                name="Actual Power"
            ))
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(
                x=df_filtered["Date_start"],
                y=df_filtered["Power_kW"],
                mode="markers",
                name="Actual Power"
            ))

        # Plot additional features if selected.
        extra_axes_layout = {}
        axis_width = 0.04
        if feature_choice and len(feature_choice) > 0:
            for i, feat in enumerate(feature_choice):
                if feat in df_filtered.columns:
                    axis_id = f"y{i + 2}"
                    layout_key = f"yaxis{i + 2}"
                    pos = 1 - ((i + 1) * axis_width)
                    if chart_type == "bar":
                        fig.add_trace(go.Bar(
                            x=df_filtered["Date_start"],
                            y=df_filtered[feat],
                            name=feat,
                            yaxis=axis_id
                        ))
                    elif chart_type == "scatter":
                        fig.add_trace(go.Scatter(
                            x=df_filtered["Date_start"],
                            y=df_filtered[feat],
                            mode="markers",
                            name=feat,
                            yaxis=axis_id
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df_filtered["Date_start"],
                            y=df_filtered[feat],
                            mode="lines",
                            name=feat,
                            line={"dash": "dot", "width": 2},
                            yaxis=axis_id
                        ))
                    extra_axes_layout[layout_key] = dict(
                        overlaying='y',
                        side='right',
                        position=pos,
                        showgrid=False,
                        title=''
                    )

        # Update graph layout with titles and margin adjustments.
        fig.update_layout(
            title="Actual Power Consumption",
            xaxis_title="Date",
            yaxis=dict(title="Power (kW)"),
            margin=dict(r=200),
            **extra_axes_layout
        )

        # Add annotation with consumption metrics.
        total_consumption_kWh = df_filtered["Power_kW"].sum()
        average_consumption_kWh = df_filtered["Power_kW"].mean()
        total_consumption_MWh = total_consumption_kWh / 1000.0
        average_consumption_MWh = average_consumption_kWh / 1000.0
        fig.add_annotation(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=(f"Total Consumption: {total_consumption_MWh:.2f} MWh | "
                  f"Average Hourly Consumption: {average_consumption_MWh:.2f} MWh (Over the selected time range)"),
            showarrow=False,
            font=dict(size=14, color="black"),
            align="center"
        )
        fig.update_layout(margin=dict(t=120))
        return fig

    # -------------------------------
    # DISPLAY MODE: COMPARISON (2019 only)
    # -------------------------------
    elif display_mode == "comparison":
        # Ensure that 2019 data is selected.
        if year_choice != "2019":
            fig = go.Figure()
            fig.add_annotation(
                text="Comparison mode is only available for 2019. Please select 2019.",
                xref="paper", yref="paper",
                showarrow=False,
                font={"color": "red"}
            )
            return fig

        # Ensure a valid model is selected.
        if not model_choice or model_choice not in ["Random Forest", "Linear Regression", "XGBoost"]:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select a regression model.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="red", size=16)
            )
            return fig

        else:
            try:
                # Load the selected model.
                if model_choice == "Random Forest":
                    model_file = "random_forest_model_jonas.pkl"
                elif model_choice == "XGBoost":
                    model_file = "xgboost_model_jonas.pkl"
                elif model_choice == "Linear Regression":
                    model_file = "linear_regression_model_jonas.pkl"
                else:
                    model_file = "random_forest_model_jonas.pkl"
                with open(model_file, "rb") as f:
                    model = pickle.load(f)

                features = ["Power_kW-1", "Power_kW-24", "solarRad_W/m2", "temp_C", "Weekday"]
                features = [feat for feat in features if feat in df_filtered.columns]
                X = df_filtered[features]
                y_pred = model.predict(X)

                # Create a comparison figure with actual and forecasted data.
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_filtered["Date_start"],
                    y=df_filtered["Power_kW"],
                    mode="lines",
                    name="Actual 2019 Power",
                    line={"width": 2}
                ))
                fig.add_trace(go.Scatter(
                    x=df_filtered["Date_start"],
                    y=y_pred,
                    mode="lines",
                    name="Forecasted 2019 Power",
                    line={"dash": "dash", "width": 2}
                ))

                # Add additional features if selected.
                extra_axes_layout = {}
                axis_width = 0.04
                if feature_choice and len(feature_choice) > 0:
                    for i, feat in enumerate(feature_choice):
                        if feat in df_filtered.columns:
                            axis_id = f"y{i + 2}"
                            layout_key = f"yaxis{i + 2}"
                            pos = 1 - ((i + 1) * axis_width)
                            fig.add_trace(go.Scatter(
                                x=df_filtered["Date_start"],
                                y=df_filtered[feat],
                                mode="lines",
                                name=feat,
                                line={"dash": "dot", "width": 2},
                                yaxis=axis_id
                            ))
                            extra_axes_layout[layout_key] = dict(
                                overlaying='y',
                                side='right',
                                position=pos,
                                showgrid=False,
                                title=''
                            )

                # Optionally, list extra features as an annotation.
                annotation_text = ", ".join(feature_choice) if feature_choice else ""
                if annotation_text:
                    fig.add_annotation(
                        x=1.0,
                        y=0.5,
                        xref='paper',
                        yref='paper',
                        text=annotation_text,
                        showarrow=False,
                        textangle=270,
                        font=dict(size=12, color='#46555f'),
                        xanchor='left',
                        yanchor='middle'
                    )

                fig.update_layout(
                    title=f"2019: Actual vs Forecasted Power Consumption using {model_choice.upper()}",
                    xaxis_title="Date",
                    yaxis=dict(title="Power (kW)"),
                    margin=dict(r=200),
                    **extra_axes_layout
                )
                return fig
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error loading model: {str(e)}",
                                   xref="paper", yref="paper", showarrow=False)
                return fig

    # Default return in case no display mode matches.
    return go.Figure()

# -------------------------------
# RUN THE APP
# -------------------------------
if __name__ == "__main__":
    app.run_server(debug=False) 
