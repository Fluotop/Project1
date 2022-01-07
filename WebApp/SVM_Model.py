import dash_table
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


Ticker = "TSLA"
Amount_of_days = "60d"
Interval_to_predict = "15m"
RSI_SMA_ADX_Period = 10
Fraction_training_data = 0.8
CVFolds = 10
df = pd.DataFrame([])
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()
cr = ""
X = []
prediction =""
def get_data():
    global df
    df = yf.download(Ticker, period = Amount_of_days, interval = Interval_to_predict)


def get_stock():
    global fig1
    get_data()
    ##STOCK FIGUUR
    #declare figure
    fig1 = go.Figure()

    #Candlestick
    fig1.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name = 'market data'))

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
    fig1.update_yaxes()
    return fig1

def get_indicators():

    global df
    #rijeen met 0-values removed
    df.isin([0]).any().any().sum()
    ##BEREKEN INDICATOREN

    #Bereken SMA en voeg kolom toe aan df
    df["SMA"] = df["Close"].shift(1).rolling(window = RSI_SMA_ADX_Period).mean()
    #Bereken corr tussen SMA en market value en voeg kolom toe aan df
    df["Corr"] = df["Close"].shift(1).rolling(window = RSI_SMA_ADX_Period).corr(df["SMA"].shift(1))

    #Bereken SAR en voeg kolom to aan df
    df["SAR"] = ta.SAR(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), 0.2, 0.2)

    #Bereken ADX en voeg kolom to aan df
    df["ADX"] = ta.ADX(np.array(df["High"].shift(1)), np.array(df["Low"].shift(1)), np.array(df["Open"]), timeperiod = RSI_SMA_ADX_Period)

    #Voeg nieuwe kolommen toe aan df
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)

    #Verschil open-prev_open en open-prev_close voor meten sensitiviteit/volatiliteit
    df["OO"] = df["Open"] - df["Open"].shift(1)
    df["OC"] = df["Open"] - df["Prev_Close"]

    #Percent winst = (next-current)/current
    df['Ret'] = (df['Open'].shift(-1)-df['Open'])/df['Open']
    # Percent winst over langere periodes
    for i in range(1, RSI_SMA_ADX_Period):
        df['return%i' % i] = df['Ret'].shift(i)

    df = df.dropna()


def build_model():
##MODEL OPSTELLEN
    global df
    global X
    global fig2
    global fig3
    global fig4
    global cr

    split = int(Fraction_training_data*len(df))

    #Negeer errors

    warnings.filterwarnings("ignore")

    #Beste 66% returns krijgen 1. Slechste 34% returns krijgen -1. Rest 0
    df["Signal"] = 0
    df.loc[df["Ret"] > df["Ret"][:split].quantile(q=0.66), "Signal"] = 1
    df.loc[df["Ret"] < df["Ret"][:split].quantile(q=0.34), "Signal"] = -1

    #Open, close, high en low reduntante informatie
    X = df.drop(["Close","Signal","High","Low","Volume","Ret"], axis=1)
    y = df["Signal"]

    #Parameters
    c = [10,100,1000,10000]
    g = [1e-3,1e-2,1e-1,1e0]
    parameters = {'svc__C': c,
                'svc__gamma': g,
                'svc__kernel': ['rbf']
                }

    steps = [('scaler', StandardScaler()), ('svc',SVC())]
    pipeline = Pipeline(steps)
    #zoek optimale parameters cv met timeseries want niet onafhankelijk
    rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=CVFolds))

    #optimale parameters op basis van training
    rcv.fit(X.iloc[:split], y.iloc[:split])
    best_c = rcv.best_params_['svc__C']
    best_gamma = rcv.best_params_['svc__gamma']
    best_kernel = rcv.best_params_['svc__kernel']
    #SVM
    cls = SVC(C=best_c, kernel = best_kernel, gamma = best_gamma)
    ss = StandardScaler()
    cls.fit(ss.fit_transform(X.iloc[:split]), y.iloc[:split])
    #predict signal
    y_predict = cls.predict(ss.transform(X.iloc[split:]))
    df["Pred_signal"] = 0

    # Save the predicted values for the train data
    df.iloc[:split, df.columns.get_loc("Pred_signal")] = pd.Series(
        cls.predict(ss.transform(X.iloc[:split])).tolist())

    # Save the predicted values for the test data
    df.iloc[split:, df.columns.get_loc("Pred_signal")] = y_predict

    # %verandering * signaal
    df["Ret1"] = df["Ret"]*df["Pred_signal"]

    ## FIGUUR RETURN
    fig2 = go.Figure()
    #cumprod percentages
    fig2.add_trace(go.Scatter(x=df.index[split:], y= (df['Ret'][split:]+1).cumprod(),line=dict(color='royalblue', width=.8), name = 'Buy and Hold Strategy'))
    fig2.add_trace(go.Scatter(x=df.index[split:], y= (df['Ret1'][split:]+1).cumprod(),line=dict(color='orange', width=.8), name = 'SVM Strategy'))



    fig2.update_layout(
        title='Test Data % Return',
        yaxis_title='Stock return (% Return)')

    #vooral hoeken checken
    cm = confusion_matrix(y[split:],y_predict)
    fig3 = go.Figure(data=[go.Table(header=dict(values=['<b>Confusion matrix</b>', 'Sell','Hold','Buy']),
                                    cells=dict(values=[['Predicted Sell', 'Predicted Hold', 'Predicted Buy'], cm[:,0],cm[:,1],cm[:,2]],
                                               fill=dict(color=['#c8d4e3', 'coral', 'coral', 'coral'])))
                           ])

    #Accuracy
    #Precision — What percent of your predictions were correct
    #Recall — What percent of the positive cases were catched
    #F1 score — What percent of positive predictions were correct
    cr = classification_report(y[split:],y_predict)
    x = cr.split()
    x1 = ["<b>Classification Report</b>"] + x[0:4]
    x2 = ["Sell"] + x[5:9]
    x3 = ["Hold"] + x[10:14]
    x4 = ["Buy"] + x[15:19]
    x5 = [""] +  [""] +[""] +[""] +[""]
    x6 = x[19:20] + [""] + [""] + x[20:22]
    x7 = [" ".join(x[22:24])] +x[24:28]
    x8 = [" ".join(x[28:30])] + x[30:34]
    fig4 = go.Figure(data=[go.Table(header=dict(values=x1),
                                cells=dict(values=np.transpose([x2,x3,x4,x5,x6,x7,x8]),
                                           fill=dict(color=['#c8d4e3', ['coral', 'coral', 'coral', '#c8d4e3','coral','coral','coral']])))
                       ])

    #Model opslaan
    joblib.dump(cls, 'model.pkl')
    return fig2, fig3, fig4

def make_prediction():
    # a = list(Interval_to_predict)
    # if a[-1] == "m":
    #     if len(a) ==3:
    #         timer = a[0]+a[1] * 1000*60
    #     elif len(a) ==2:
    #         timer = a[0] * 1000*60
    # elif a[-1] == "h":
    #     if len(a) ==2:
    #         timer = a[0] * 1000*600
    # elif a[-1] == "d":
    #     if len(a) ==2:
    #         timer = a[0] * 8.64e+7


    global prediction
    #build_model()
    cls = joblib.load('model.pkl')
    query = X.iloc[-1]
    prediction = cls.predict([query])
    return str(prediction)

# get_data()
#
# get_indicators()
# build_model()
# make_prediction()
#TODO dfcopy iedere keer als df veranderd