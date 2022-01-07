# Project1

DATA
Originates from the Yahhoo Finance API

For a selected timeperiod the following variables are available for a selected time interval:
RAW (used as is)
  Open (O): The opening price of the stock
  Close (C): The closing price of the stock
  High (H): The highest price of the stock in the interval
  Low (L): The lowest price of the stock in the interval
INDICATORS (Calculated based on raw variables)
  Revative Strength Index (RSI): Detects overbought and undersold conditions
  Simple Moving Average (SMA): The average closing price over a selected time period
  Parabolic Stop And Reverse (SAR): Detects trend direction
  Average Directional Index (ADX): Estimates trend strength
  Raw variables of T-1: The OCHL variables of the previous timepoint. 
  
MODEL
Variables used: 
  Open
  Close
  High
  Low
