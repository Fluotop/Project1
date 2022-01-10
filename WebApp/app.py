import time
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import SVM_Model
from SVM_Model import *

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.COSMO])

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
                    value="TSLA"
                ),
                html.Br(),
                html.Label('Days used to predict:'),
                dcc.Input(
                    id="Amount_of_days",
                    type="text",
                    placeholder="xxxxd(ays)",
                    value="60d",
                    style={'width': '9.5%'},
                ),
                html.Br(),
                html.Label('Predict every:'),
                dcc.Input(
                    id="Interval_to_predict",
                    type="text",
                    placeholder="1,2,5,15,30,60,90m,1d or 5d",
                    value="5m"
                ),
                html.Br(),
                html.Button(
                    "Fetch Stock",
                    id="Fetch_stock"
                ),
             ]),
            html.Br(),
            dbc.Col(dcc.Graph(id="fig1", className= "shadow-lg",style={'width': '90%'}))

        ]),


        dcc.Tab(label='Simulation', value='simulationtab', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('#Datapoints RSI/SMA/ADX: '),
                    dcc.Input(
                        id="RSI_SMA_ADX_Period",
                        type="number",
                        placeholder="Default 10",
                        value=10
                    ),
                 ]),
            dbc.Row([
                dbc.Col([
                    html.Label('Fraction of data used for training: '),
                    dcc.Input(
                        id="Fraction_training_data",
                        type="number",
                        placeholder="Default 0.8",
                        value=0.8
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
                            value=10
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
                    style={"width":800, "margin": 0, 'display': 'inline-block'},

                ),
                html.Div(
                    dcc.Graph(id='fig3',
                              ),
                    className="six columns",
                    style={"width":450, "margin": 0, 'display': 'inline-block'}
                ),
            ], className="row"),

            html.Div([
                html.Div(
                    dcc.Graph(id='fig4',
                              ),
                    className="six columns",
                    style={"width":900, "margin": 0, 'display': 'inline-block'},

                ),

                html.Div(
                    html.Div(id = "time"),
                    className="six columns",
                    style={"width":300, "margin": 0, 'display': 'inline-block'}
                ),
            ], className="row"),
        ]),

        dcc.Tab(label='Results', value='resulttab', children= [
            html.Button(
                "Get current prediction",
                id="Make_Prediction"
            ),
            html.Div(
                id = "prediction"
            )
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
def update_settings(Fetch_stock, Ticker, Amount_of_days, Interval_to_predict):
    SVM_Model.Ticker = Ticker
    SVM_Model.Amount_of_days = Amount_of_days
    SVM_Model.Interval_to_predict = Interval_to_predict
    get_stock()
    return SVM_Model.fig1


@app.callback(
    Output('fig2', 'figure'),
    Output('fig3', 'figure'),
    Output('fig4', 'figure'),
    Output('time',"children"),
    Input("Build_Model", "n_clicks"),
    State('RSI_SMA_ADX_Period', 'value'),
    State('Fraction_training_data', 'value'),
    State('CVFolds', 'value'),
    prevent_initial_call=True
)
def update_simulation(Build_Model, RSI_SMA_ADX_Period, Fraction_training_data, CVFolds):
    SVM_Model.RSI_SMA_ADX_Period = RSI_SMA_ADX_Period
    SVM_Model.Fraction_training_data = Fraction_training_data
    SVM_Model.CVFolds = CVFolds
    t0 = time.time()
    get_data()
    get_indicators()
    build_model()
    t1 = time.time()
    #plus one to make sure the prediction function can be included in this time
    total = str(round((t1 - t0) + 1, 2))
    timestring = "It takes approximately "+total+"sec to make a prediction"
    return SVM_Model.fig2, SVM_Model.fig3, SVM_Model.fig4, timestring

@app.callback(
    Output('prediction', 'children'),
    Input("Make_Prediction", "n_clicks"),
    prevent_initial_call=True
)
def update_Prediction(Make_Prediction):
    #get_data()
    #get_indicators()
    #build_model()
    make_prediction()
    return SVM_Model.prediction

if __name__ == '__main__':
    app.run_server(debug=False)
