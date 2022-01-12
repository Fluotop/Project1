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
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
import talib as ta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import warnings

# Packages used in the App
import time
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

#############################################################################################################
##	Constants with their default values. These are free to be changed                				       ##
#############################################################################################################

Ticker = "TSLA"
Amount_of_days = "60d"
Interval_to_predict = "15m"
RSI_SMA_ADX_Period = 10
Fraction_training_data = 0.8
CVFolds = 10


#############################################################################################################

# Downloads and initiates the stock dataframe
def get_data(Ticker, Amount_of_days, Interval_to_predict):
    df = yf.download(Ticker, period=Amount_of_days, interval=Interval_to_predict)
    return df


def get_stock(df):
    ##STOCK FIGUUR
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


def get_indicators(df, RSI_SMA_ADX_Period):
    # rijeen met 0-values removed
    df.isin([0]).any().any().sum()
    ##BEREKEN INDICATOREN

    # Bereken SMA en voeg kolom toe aan df
    df["SMA"] = df["Close"].shift(1).rolling(window=RSI_SMA_ADX_Period).mean()
    # Bereken corr tussen SMA en market value en voeg kolom toe aan df
    df["Corr"] = df["Close"].shift(1).rolling(window=RSI_SMA_ADX_Period).corr(df["SMA"].shift(1))

    # Bereken SAR en voeg kolom to aan df
    df["SAR"] = ta.SAR(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), 0.2, 0.2)

    # Bereken ADX en voeg kolom to aan df
    df["ADX"] = ta.ADX(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), np.array(df["Open"]),
                       timeperiod=RSI_SMA_ADX_Period)

    # Voeg nieuwe kolommen toe aan df
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)

    # Verschil open-prev_open en open-prev_close voor meten sensitiviteit/volatiliteit
    df["OO"] = df["Open"] - df["Open"].shift(1)
    df["OC"] = df["Open"] - df["Prev_Close"]

    # Percent winst = (next-current)/current
    df['Ret'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    # Percent winst over langere periodes
    for i in range(1, RSI_SMA_ADX_Period):
        df['return%i' % i] = df['Ret'].shift(i)

    df = df.dropna()
    return df


def build_model(df, Fraction_training_data, CVFolds):
    ##MODEL OPSTELLEN

    split = int(Fraction_training_data * len(df))

    # Negeer errors

    warnings.filterwarnings("ignore")

    # Beste 66% returns krijgen 1. Slechste 34% returns krijgen -1. Rest 0
    df["Signal"] = 0
    df.loc[df["Ret"] > df["Ret"][:split].quantile(q=0.66), "Signal"] = 1
    df.loc[df["Ret"] < df["Ret"][:split].quantile(q=0.34), "Signal"] = -1

    # Open, close, high en low reduntante informatie
    X = df.drop(["Close", "Signal", "High", "Low", "Volume", "Ret"], axis=1)
    y = df["Signal"]

    # Parameters
    c = [10, 100, 1000, 10000]
    g = [1e-3, 1e-2, 1e-1, 1e0]
    parameters = {'svc__C': c,
                  'svc__gamma': g,
                  'svc__kernel': ['rbf']
                  }

    steps = [('scaler', StandardScaler()), ('svc', SVC())]
    pipeline = Pipeline(steps)
    # zoek optimale parameters cv met timeseries want niet onafhankelijk
    rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=CVFolds))

    # optimale parameters op basis van training
    rcv.fit(X.iloc[:split], y.iloc[:split])
    best_c = rcv.best_params_['svc__C']
    best_gamma = rcv.best_params_['svc__gamma']
    best_kernel = rcv.best_params_['svc__kernel']
    # SVM
    cls = SVC(C=best_c, kernel=best_kernel, gamma=best_gamma)
    ss = StandardScaler()
    cls.fit(ss.fit_transform(X.iloc[:split]), y.iloc[:split])
    # predict signal
    y_predict = cls.predict(ss.transform(X.iloc[split:]))
    df["Pred_signal"] = 0

    # Save the predicted values for the train data
    df.iloc[:split, df.columns.get_loc("Pred_signal")] = pd.Series(
        cls.predict(ss.transform(X.iloc[:split])).tolist())

    # Save the predicted values for the test data
    df.iloc[split:, df.columns.get_loc("Pred_signal")] = y_predict

    # %verandering * signaal
    df["Ret1"] = df["Ret"] * df["Pred_signal"]

    ## FIGUUR RETURN
    fig2 = go.Figure()
    # cumprod percentages
    fig2.add_trace(
        go.Scatter(x=df.index[split:], y=(df['Ret'][split:] + 1).cumprod(), line=dict(color='royalblue', width=.8),
                   name='Buy and Hold Strategy'))
    fig2.add_trace(
        go.Scatter(x=df.index[split:], y=(df['Ret1'][split:] + 1).cumprod(), line=dict(color='orange', width=.8),
                   name='SVM Strategy'))

    fig2.update_layout(
        title='Test Data % Return',
        yaxis_title='Stock return (% Return)')

    # vooral hoeken checken
    cm = confusion_matrix(y[split:], y_predict)
    fig3 = go.Figure(data=[go.Table(header=dict(values=['<b>Confusion matrix</b>', 'Sell', 'Hold', 'Buy']),
                                    cells=dict(values=[['Predicted Sell', 'Predicted Hold', 'Predicted Buy'], cm[:, 0],
                                                       cm[:, 1], cm[:, 2]],
                                               fill=dict(color=['#c8d4e3', 'coral', 'coral', 'coral'])))
                           ])

    # Accuracy
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

    # Model opslaan
    joblib.dump(cls, 'model.pkl')
    query = X.iloc[-1]
    prediction = cls.predict([query])
    return fig2, fig3, fig4, prediction


def make_prediction(Interval_to_predict):
    # Wait until the start of the next interval
    current_time = round(time.time())
    waittime = int(Interval_to_predict[:-1]) * 60
    time.sleep(waittime - current_time % waittime)

    prediction = build_model(
        get_indicators(
            get_data(Ticker, Amount_of_days, Interval_to_predict
                     ), RSI_SMA_ADX_Period
        ), Fraction_training_data, CVFolds
    )[3]
    prediction = str(prediction[0])
    return prediction


# make_prediction(Interval_to_predict)


# get_data()
#
# get_indicators()
# build_model()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.H1('SVM Day Trading Bot V0.1'),
    html.H5('Model and app created by Ben De Maesschalck'),
    dcc.Tabs(id="tabs", value='tabs', children=[
        dcc.Tab(label='Settings', value='settingstab', children=[
            html.Br(),
            html.Div(children=[
                html.Label('Stock Ticker:'),
                dcc.Input(
                    id="Ticker",
                    type="text",
                    placeholder="Yahoo Finance Stock Ticker",
                    value=Ticker
                ),
                html.Br(),
                html.Label('Days used to predict:'),
                dcc.Input(
                    id="Amount_of_days",
                    type="text",
                    placeholder="xxxxd(ays)",
                    value=Amount_of_days,
                    style={'width': '9.5%'},
                ),
                html.Br(),
                html.Label('Predict every:'),
                dcc.Input(
                    id="Interval_to_predict",
                    type="text",
                    placeholder="1,2,5,15,30,60,90m,1d or 5d",
                    value=Interval_to_predict
                ),
                html.Br(),
                html.Button(
                    "Fetch Stock",
                    id="Fetch_stock"
                ),
            ]),
            html.Br(),
            dbc.Col(dcc.Graph(id="fig1", className="shadow-lg", style={'width': '90%'}))

        ]),

        dcc.Tab(label='Simulation', value='simulationtab', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('#Datapoints RSI/SMA/ADX: '),
                    dcc.Input(
                        id="RSI_SMA_ADX_Period",
                        type="number",
                        placeholder="Default 10",
                        value=RSI_SMA_ADX_Period
                    ),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label('Fraction of data used for training: '),
                        dcc.Input(
                            id="Fraction_training_data",
                            type="number",
                            placeholder="Default 0.8",
                            value=Fraction_training_data
                        ),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label('# Fold Crossvalidation '),
                        dcc.Input(
                            id="CVFolds",
                            type="number",
                            placeholder="Default 10",
                            value=CVFolds
                        ),
                    ]),
                ]),
                dbc.Col([
                    html.Button(
                        "Build Model",
                        id="Build_Model"
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

        dcc.Tab(label='Results', value='resulttab', children=[
            html.Button(
                "Get current prediction",
                id="Make_Prediction"
            ),
            html.Button(
                "Stop predicting",
                id="Stop_Prediction"
            ),
            html.Div(id='prediction'
                     ),
            dcc.Interval(id = "trigger",
                         interval= int(Interval_to_predict[:-1]) * 6000)
        ]),

    ]),
])


@app.callback(
    Output('fig1', 'figure'),
    Input("Fetch_stock", "n_clicks"),
    State('Ticker', 'value'),
    State('Amount_of_days', 'value'),
    State('Interval_to_predict', 'value'),
    prevent_initial_call=True
)
def update_settings(Fetch_stock, Ticker_new, Amount_of_days_new, Interval_to_predict_new):
    global Ticker
    global Amount_of_days
    global Interval_to_predict

    Ticker = Ticker_new
    Amount_of_days = Amount_of_days_new
    Interval_to_predict = Interval_to_predict_new
    fig1 = get_stock(
        get_data(Ticker, Amount_of_days, Interval_to_predict)
    )
    return fig1


@app.callback(
    Output('fig2', 'figure'),
    Output('fig3', 'figure'),
    Output('fig4', 'figure'),
    Output('time', "children"),
    Input("Build_Model", "n_clicks"),
    State('RSI_SMA_ADX_Period', 'value'),
    State('Fraction_training_data', 'value'),
    State('CVFolds', 'value'),
    State('Ticker', 'value'),
    State('Amount_of_days', 'value'),
    State('Interval_to_predict', 'value'),
    prevent_initial_call=True
)
def update_simulation(Build_Model, RSI_SMA_ADX_Period, Fraction_training_data, CVFolds, Ticker, Amount_of_days,
                      Interval_to_predict):
    t0 = time.time()
    [fig2, fig3, fig4] = build_model(
        get_indicators(
            get_data(Ticker, Amount_of_days, Interval_to_predict
                     ), RSI_SMA_ADX_Period
        ), Fraction_training_data, CVFolds
    )[0:3]
    t1 = time.time()
    # plus one to make sure the prediction function can be included in this time
    total = str(round((t1 - t0) + 1, 2))
    timestring = "It takes approximately " + total + "sec to make a prediction"
    return fig2, fig3, fig4, timestring


@app.callback(
    Output('prediction', 'children'),
    Output("Stop_Prediction", "n_clicks"),
    Input("Make_Prediction", "n_clicks"),
    Input("Stop_Prediction", "n_clicks"),
    Input("trigger", "n_intervals"),
    prevent_initial_call=True
)
def update_Prediction(Make_Prediction, Stop_Prediction, trigger):
    prediction_list = ""
    if Stop_Prediction == None:
        prediction = make_prediction(Interval_to_predict)
        now = datetime.now()
        timestr = now.strftime("%d/%m/%Y %H:%M:%S")
        prediction_string = timestr + prediction
        prediction_list = prediction_list + prediction_string
        return prediction_list, Stop_Prediction
    prediction_list = prediction_list + "It's over don't you get it"
    Stop_Prediction = None
    return prediction_list, Stop_Prediction


if __name__ == '__main__':
    app.run_server(debug=True)
