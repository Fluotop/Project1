"""""
;===================================================================================================
; Title:   Support vector machine model for day trading
; Author: Ben De Maesschalck
;===================================================================================================
Dependencies to install through pip or conda:
- pandas
- numpy
- yfinance
- talib
- sklearn.metrics
- sklearn.pipeline
- sklearn.model_selection
- sklearn.preprocessing
- sklearn.svm
- dash
- dash-bootstrap-components
- dash-core-components
- dash_html_components
"""
# Packages used in the Model


# Packages used in the App
import time
import warnings
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import talib as ta
import yfinance as yf
from dash.dependencies import Input, Output, State
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#############################################################################################################
##	Constants with their default values. These are free to be changed                				       ##
#############################################################################################################

TICKER = "TSLA"
AMOUNT_OF_DAYS = "5d"
INTERVAL_TO_PREDICT = "2m"
RSI_SMA_ADX_PERIOD = 10
FRACTION_TRAINING_DATA = 0.8
CV_FOLDS = 2

prediction_list = ""


#############################################################################################################

#
def get_data(ticker, amount_of_days, interval_to_predict):
    """Downloads stock data corresponding to the given ticker. The dataset spans X previous days and contains info
    about every X minutes

    :parameter
    ticker: str
        The ticker name of the stock on https://finance.yahoo.com/. (default is TSLA)
    amount_of_days: str
        The amount of days into the past. (default is 60d)
    interval_to_predict: str
        The interval at which data is available each day. (default is 15m)

    :returns
    pandas.core.frame.DataFrame
        A pandas dataframe containing the open, close, high, low and adj. close price of the chosen stock and timerange.
    """

    df = yf.download(ticker, period=amount_of_days, interval=interval_to_predict)
    return df


def get_stock(df):
    """Visualises the downloaded data from the get_data() method.

    :parameter
    df: pandas.core.frame.DataFrame
        Stock data downloaded from Yahoo Finance by get_data()

    :returns
    fig1: plotly.graph_objs._figure.Figure
        A figure containing a candlestick graph with adjustable X-axis.
    """

    # declare figure
    fig1 = go.Figure()

    # Candlestick
    fig1.add_trace(go.Candlestick(x=df.index,
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'], name='market data'))

    # Add titles
    fig1.update_layout(
        title='Stock live share price evolution',
        yaxis_title='Stock Price (USD per Shares)')

    # X-Axes
    fig1.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig1


def get_indicators(df, window):
    """Enriches the downloaded stock dataframe from get_data() by adding the following derivatives and indicators:

    Relative Strength Index (RSI): Detects overbought and undersold conditions {float}
    Simple Moving Average (SMA): The average closing price over a selected time period {float}
    Corr: The correlation between the closing price (C) and the Simple Moving Average (SMA) {float64}
    Parabolic Stop And Reverse (SAR): Detects trend direction {float}
    Average Directional Index (ADX): Estimates trend strength {float}
    Raw variables of T-1: The OCHL variables of the previous time-point {float64}
    Opening difference (OO): Difference between the current open price (O) and the open price (O) at T-1 {float}
    Open close difference (OC): Difference between the current open price (O) and the close price (C) at T-1 {float}
    Ret: Percent gain in open price compared to the previous (T-1) time periods {float}
    Return(X=1-10): Percent gain in open price at time point T-X {float}


    :parameter
    df: pandas.core.frame.DataFrame
        Stock data downloaded from Yahoo Finance by get_data().
    window: int
        Timespan over which part of the derivatives and indicators are calculated.

   :returns
    df_complete: pandas.core.frame.DataFrame
        A pandas dataframe containing 23 variables: the open, close, high, low, adj close, derivatives and indicators.
    """

    # Rows with 0-values removed
    df.isin([0]).any().any().sum()
    # Indicators

    # Calculate SMA and add column to df
    df["SMA"] = df["Close"].shift(1).rolling(window=window).mean()
    # Calculate Corr between SMA and market close and add column to df
    df["Corr"] = df["Close"].shift(1).rolling(window=window).corr(df["SMA"].shift(1))

    # Calculate SAR and add column to df
    df["SAR"] = ta.SAR(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), 0.2, 0.2)

    # Calculate ADX and add column to df
    df["ADX"] = ta.ADX(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), np.array(df["Open"]),
                       timeperiod=window)

    # Add columns containing stock data from the previous day
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)

    # Calculate the difference between today's open and previous open and also the difference between today's close and
    # previous days close. Add columns to df. (Used to measure sensitivity and volatility)
    df["OO"] = df["Open"] - df["Open"].shift(1)
    df["OC"] = df["Open"] - df["Prev_Close"]

    #  Calculate the return as ret = (next-current)/current and add column to df.
    df['Ret'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    # Calculate return compared up to 10 days ago. Add the 10 columns to the df
    for i in range(1, RSI_SMA_ADX_PERIOD):
        df['return%i' % i] = df['Ret'].shift(i)

    # Remove rows that are incomplete because no data about previous days is available.
    df_complete = df.dropna()
    return df_complete


def build_model(df_complete, fraction_training_data, cv_folds):
    """Create an SVM model on the supplied data. Labels are added to the data based on return: the 34% highest returns
    are classified as buy(1), 33% lowest as sell(-1) rest as hold(0). SVM parameters are optimised via a gridsearch.
    The performance of the model is assessed based on returns, the confusionmatrix and the classification report.
    This function stores the created model locally as a .pkl file.

    :parameter
    df_complete: pandas.core.frame.DataFrame
        Stock data containing all 23 parameters to build the SVM model. Generated by get_indicators().
    fraction_training_data: float
        The fraction of data in df_complete that is used to train the SVM model.
        1-fraction_training_data will be used in testing.
    cv_folds: int
        The amount of X-fold cross validation to use when building the model.
    :returns
    fig2: plotly.graph_objs._figure.Figure
        A figure containing the returns on the test data of the SVM vs the returns of a buy and hold strategy.
    fig3: plotly.graph_objs._figure.Figure
        A confusion matrix of the SVM model's decisions on the test data
    fig4: plotly.graph_objs._figure.Figure
        The classification report of the SVM
    prediction: int
        The prediction for the next timepoint (1 for buy, 0 for hold and -1 for sell)
    """

    # Number of rows to use for training
    split = int(fraction_training_data * len(df_complete))

    # Ignore error that does not affect the code.
    warnings.filterwarnings("ignore")

    # Best 34% returns are labeled 1, the worst 33% returns as -1, rest as 0 in a new column: Signal.
    df_complete["Signal"] = 0
    df_complete.loc[df_complete["Ret"] > df_complete["Ret"][:split].quantile(q=0.66), "Signal"] = 1
    df_complete.loc[df_complete["Ret"] < df_complete["Ret"][:split].quantile(q=0.34), "Signal"] = -1

    # Drop columns with redundant correlated information and store labels separately.
    X = df_complete.drop(["Close", "Signal", "High", "Low", "Volume", "Ret"], axis=1)
    y = df_complete["Signal"]

    # Search the best parameters for the model via gridsearch and a special CV technique for timeseries data.
    c = [10, 100, 1000, 10000]
    g = [1e-3, 1e-2, 1e-1, 1e0]
    parameters = {'svc__C': c,
                  'svc__gamma': g,
                  'svc__kernel': ['rbf']
                  }

    steps = [('scaler', StandardScaler()), ('svc', SVC())]
    pipeline = Pipeline(steps)
    rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=cv_folds))

    # Build model on training data based on optimal parameters
    rcv.fit(X.iloc[:split], y.iloc[:split])
    best_c = rcv.best_params_['svc__C']
    best_gamma = rcv.best_params_['svc__gamma']
    best_kernel = rcv.best_params_['svc__kernel']
    cls = SVC(C=best_c, kernel=best_kernel, gamma=best_gamma)
    ss = StandardScaler()
    cls.fit(ss.fit_transform(X.iloc[:split]), y.iloc[:split])
    # Predict labels for the test data and store them in new column
    y_predict = cls.predict(ss.transform(X.iloc[split:]))
    df_complete["Pred_signal"] = 0

    # Save the predicted values for the train data
    df_complete.iloc[:split, df_complete.columns.get_loc("Pred_signal")] = pd.Series(
        cls.predict(ss.transform(X.iloc[:split])).tolist())

    # Save the predicted values for the test data
    df_complete.iloc[split:, df_complete.columns.get_loc("Pred_signal")] = y_predict

    # Calculate the return of the models decision as the return * the signal and store in a new column
    df_complete["Ret1"] = df_complete["Ret"] * df_complete["Pred_signal"]

    # Create the return figure
    fig2 = go.Figure()
    # cumprod percentages
    fig2.add_trace(
        go.Scatter(x=df_complete.index[split:], y=(df_complete['Ret'][split:] + 1).cumprod(),
                   line=dict(color='royalblue', width=.8),
                   name='Buy and Hold Strategy'))
    fig2.add_trace(
        go.Scatter(x=df_complete.index[split:], y=(df_complete['Ret1'][split:] + 1).cumprod(),
                   line=dict(color='orange', width=.8),
                   name='SVM Strategy'))

    fig2.update_layout(
        title='Test Data % Return',
        yaxis_title='Stock return (% Return)')

    # Create the confusion matrix
    cm = confusion_matrix(y[split:], y_predict)
    fig3 = go.Figure(data=[go.Table(header=dict(values=['<b>Confusion matrix</b>', 'Sell', 'Hold', 'Buy']),
                                    cells=dict(values=[['Predicted Sell', 'Predicted Hold', 'Predicted Buy'], cm[:, 0],
                                                       cm[:, 1], cm[:, 2]],
                                               fill=dict(color=['#c8d4e3', 'coral', 'coral', 'coral'])))
                           ])

    # Create the classification report. Contains the following info:
    # Precision — What percent of your predictions were correct
    # Recall — What percent of the positive cases were catched
    # F1 score — What percent of positive predictions were correct
    cr = classification_report(y[split:], y_predict)
    x = cr.split()
    x1 = ["<b>Classification Report</b>"] + x[0:4]
    x2 = ["Sell"] + x[5:9]
    x3 = ["Hold"] + x[10:14]
    x4 = ["Buy"] + x[15:19]
    x5 = [""] + [""] + [""] + [""] + [""]
    x6 = x[19:20] + [""] + [""] + x[20:22]
    x7 = [" ".join(x[22:24])] + x[24:28]
    x8 = [" ".join(x[28:30])] + x[30:34]
    fig4 = go.Figure(data=[go.Table(header=dict(values=x1),
                                    cells=dict(values=np.transpose([x2, x3, x4, x5, x6, x7, x8]),
                                               fill=dict(color=['#c8d4e3',
                                                                ['coral', 'coral', 'coral', '#c8d4e3', 'coral', 'coral',
                                                                 'coral']])))
                           ])

    # Store the created model and explicitly return the prediction for the most recent time-point.
    joblib.dump(cls, 'model.pkl')
    query = X.iloc[-1]
    prediction = cls.predict([query])

    return fig2, fig3, fig4, prediction


def make_prediction(interval_to_predict):
    """When called sleeps until new data is available on Yahoo Finance.
    At this moment a prediction for this new time-point is returned.

    :parameter
    interval_to_predict: str
        Interval at which new predictions have to be returned.

    :returns
    int
        The prediction for the current time-point(buy:1, hold:0 and sell:-1).
    """

    # Wait until the start of the next interval
    current_time = round(time.time())
    wait_time = int(interval_to_predict[:-1]) * 60
    time.sleep(wait_time - current_time % wait_time)
    # Get current prediction
    prediction = build_model(
        get_indicators(
            get_data(TICKER, AMOUNT_OF_DAYS, INTERVAL_TO_PREDICT
                     ), RSI_SMA_ADX_PERIOD
        ), FRACTION_TRAINING_DATA, CV_FOLDS
    )[3]
    prediction = prediction[0]
    return prediction


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
"""
App consists of a header and three panels(tabs). Panels should be used from left to right. 
First panel(Settings) selects the stock and visualizes the selected data.
Second panel(Simulation) builds the model and visualises its performance on the test set.
Third panel(Results) Prints a new prediction each time new data is available.
"""

app.layout = html.Div([


    html.H1('SVM Day Trading Bot V0.1'),
    html.H5('Model and app created by Ben De Maesschalck'),

    dcc.Tabs(id="tabs", value='tabs', children=[
        # SETTINGS TAB
        dcc.Tab(label='Settings', value='settingstab', children=[
            html.Br(),
            html.Div(children=[
                html.Label('Stock Ticker:'),
                dcc.Input(
                    id="ticker",
                    type="text",
                    placeholder="Yahoo Finance Stock Ticker",
                    value=TICKER
                ),
                html.Br(),
                html.Label('Days used to predict:'),
                dcc.Input(
                    id="amount_of_days",
                    type="text",
                    placeholder="xxxxd(ays)",
                    value=AMOUNT_OF_DAYS,
                    style={'width': '9.5%'},
                ),
                html.Br(),
                html.Label('Predict every:'),
                dcc.Input(
                    id="interval_to_predict",
                    type="text",
                    placeholder="1,2,5,15,30,60,90m,1d or 5d",
                    value=INTERVAL_TO_PREDICT
                ),
                html.Br(),
                html.Button(
                    "Fetch Stock",
                    id="fetch_stock"
                ),
            ]),
            html.Br(),
            dbc.Col(dcc.Graph(id="fig1", className="shadow-lg", style={'width': '90%'}))

        ]),
        # SIMULATION TAB
        dcc.Tab(label='Simulation', value='simulationtab', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('#Datapoints RSI/SMA/ADX: '),
                    dcc.Input(
                        id="window",
                        type="number",
                        placeholder="Default 10",
                        value=RSI_SMA_ADX_PERIOD
                    ),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label('Fraction of data used for training: '),
                        dcc.Input(
                            id="fraction_training_data",
                            type="number",
                            placeholder="Default 0.8",
                            value=FRACTION_TRAINING_DATA
                        ),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label('# Fold Crossvalidation '),
                        dcc.Input(
                            id="cv_folds",
                            type="number",
                            placeholder="Default 10",
                            value=CV_FOLDS
                        ),
                    ]),
                ]),
                dbc.Col([
                    html.Button(
                        "Build Model",
                        id="generate_model"
                    ),
                ]),
            ]),

            html.Div([
                html.Div(
                    dcc.Graph(id='fig2',
                              ),
                    className="six columns",
                    style={"width": 800, "margin": 0, 'display': 'inline-block'},

                ),
                html.Div(
                    dcc.Graph(id='fig3',
                              ),
                    className="six columns",
                    style={"width": 450, "margin": 0, 'display': 'inline-block'}
                ),
            ], className="row"),

            html.Div([
                html.Div(
                    dcc.Graph(id='fig4',
                              ),
                    className="six columns",
                    style={"width": 900, "margin": 0, 'display': 'inline-block'},

                ),
                html.Div(
                    html.Div(id="time"),
                    className="six columns",
                    style={"width": 300, "margin": 0, 'display': 'inline-block'}
                ),
            ], className="row"),
        ]),

        # RESULTS TAB
        dcc.Tab(label='Results', value='resulttab', children=[
            html.Button(
                "Get current prediction",
                id="start_predicting"
            ),
            html.Button(
                "Stop predicting",
                id="stop_predicting",
                disabled=True
            ),
            html.P(id='prediction'
                   ),
            dcc.Interval(id="trigger",
                         interval=int(INTERVAL_TO_PREDICT[:-1]) * 60000,
                         max_intervals=0),
        ]),

    ]),
])


@app.callback(
    Output('fig1', 'figure'),
    Output('trigger', "interval"),
    Input("fetch_stock", "n_clicks"),
    State('ticker', 'value'),
    State('amount_of_days', 'value'),
    State('interval_to_predict', 'value'),
    prevent_initial_call=True
)
def update_settings(fetch_stock, ticker, amount_of_days, interval_to_predict):
    """Updates data selection parameters and fetches/updates the candlestick figure of the selected data.

    :parameter
    ticker: str
        The new ticker name of the stock on https://finance.yahoo.com/. (default is TSLA)
    amount_of_days: str
        The new amount of previous data to download (in days). (default is 60d)
    interval_to_predict: str
        The new interval at which data is available each day. (default is 15m)

    :returns
     fig1: plotly.graph_objs._figure.Figure
        A figure containing a candlestick graph with adjustable X-axis.
    interval: int
        The amount of milliseconds between calls of the update_prediction() method (see below).
    """

    global TICKER
    global AMOUNT_OF_DAYS
    global INTERVAL_TO_PREDICT
    # Updates global variables with user input.
    TICKER = ticker
    AMOUNT_OF_DAYS = amount_of_days
    INTERVAL_TO_PREDICT = interval_to_predict
    # Convert interval sting (in minutes) into an int (in milliseconds).
    interval = int(INTERVAL_TO_PREDICT[:-1]) * 60000
    # Fetch candlestick figure.
    fig1 = get_stock(
        get_data(TICKER, AMOUNT_OF_DAYS, INTERVAL_TO_PREDICT)
    )

    return fig1, interval


@app.callback(
    Output('fig2', 'figure'),
    Output('fig3', 'figure'),
    Output('fig4', 'figure'),
    Output('time', "children"),
    Input("generate_model", "n_clicks"),
    State('window', 'value'),
    State('fraction_training_data', 'value'),
    State('cv_folds', 'value'),
    State('ticker', 'value'),
    State('amount_of_days', 'value'),
    State('interval_to_predict', 'value'),
    prevent_initial_call=True
)
def update_simulation(generate_model, window, fraction_training_data, cv_folds, ticker, amount_of_days,
                      interval_to_predict):
    """Based on the downloaded data, fetch an SVM model and fetch figures to assess its performance.

    :parameter
    window: int
        Timespan over which part of the derivatives and indicators are calculated.
    fraction_training_data: float
        The fraction of data in df_complete that is used to train the SVM model.
        1-fraction_training_data will be used in testing.
    cv_folds: int
        The amount of X-fold cross validation to use when building the model.
    ticker: str
        The new ticker name of the stock on https://finance.yahoo.com/.
    amount_of_days: str
        The new amount of previous data to download (in days).
    interval_to_predict: str
        The new interval at which data is available each day.

    :returns
     fig2: plotly.graph_objs._figure.Figure
        A figure containing the returns on the test data of the SVM vs the returns of a buy and hold strategy.
     fig3: plotly.graph_objs._figure.Figure
        A confusion matrix of the SVM model's decisions on the test data
     fig4: plotly.graph_objs._figure.Figure
        The classification report of the SVM
     timestring: str
        String containing the time needed to make a model and do a prediction.
    """

    # Time the code and pass all variables needed to create a model using build_model().
    t0 = time.time()
    [fig2, fig3, fig4] = build_model(
        get_indicators(
            get_data(ticker, amount_of_days, interval_to_predict
                     ), window
        ), fraction_training_data, cv_folds
    )[0:3]
    t1 = time.time()
    # plus one to make sure the prediction function can be included in this time
    total = str(round((t1 - t0) + 1, 2))
    timestring = "It takes approximately " + total + "sec to make a prediction"

    return fig2, fig3, fig4, timestring


@app.callback(
    Output('prediction', 'children'),
    Input("trigger", "n_intervals"),
    prevent_initial_call=True
)
def update_prediction(trigger):
    """Fetches the latest predictions and stores it together with the date and time in a list.
    This list grows each time this function is called.

    :parameter

    :returns
    prediction_list: list
        List storing the date and time of each made prediction. Displayed on the results tab.
    """

    global prediction_list
    prediction = make_prediction(INTERVAL_TO_PREDICT)
    now = datetime.now()
    timestr = now.strftime("%d_%m_%Y %H_%M_%S")
    prediction_string = (timestr + " " + str(prediction) + "\n")
    prediction_list = prediction_list + prediction_string
    print(prediction_list)

    return prediction_list


@app.callback(
    Output("start_predicting", "disabled"),
    Output("stop_predicting", "disabled"),
    Output("trigger", "max_intervals"),
    Output("start_predicting", "n_clicks"),
    Output("stop_predicting", "n_clicks"),
    Input("start_predicting", "n_clicks"),
    Input("stop_predicting", "n_clicks"),
    prevent_initial_call=True
)
def toggle_prediction_mode(start_predicting, stop_predicting):
    """Enables and disables automatic predicting via a start and stop button.
    When stopping predictions a text file containing all predictions is saved locally.

    :parameter
    start_predicting: int
        Amount of times the start button is pressed. Used to detect click.
    stop_predicting: int
        Amount of times the stop button is pressed. Used to detect click.

    :returns
    boolean
        Availability of the start button.
    boolean
        Availability of the stop button.
    int
        Activate of deactivate automatic prediction making at certain intervals.
    start_predicting: int
        reset the start button to not clicked.
    stop_predicting: int
        reset the stop button to not clicked.
    """
    global prediction_list
    # Enable automatic predicting
    if start_predicting is not None:
        start_predicting = None
        return True, False, -1, start_predicting, stop_predicting
    # Disable automatic predicting and write results to results.txt
    if stop_predicting is not None:
        stop_predicting = None
        text_file = open("results", "w")
        text_file.write(prediction_list)
        text_file.close()
        prediction_list = ""
        return False, True, 0, start_predicting, stop_predicting

if __name__ == '__main__':
    app.run_server(debug=True)
