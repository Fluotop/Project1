# Project1

DATA
Originates from the Yahhoo Finance API

For a selected timeperiod the following variables are available for a selected time interval:
RAW (used as is)
  Open (O): The opening price of the stock {float64}
  Close (C): The closing price of the stock {float64}
  High (H): The highest price of the stock in the interval {float64} 
  Low (L): The lowest price of the stock in the interval {float64}
INDICATORS AND DERIVATIVES (Calculated based on raw variables)
  Revative Strength Index [(RSI)](https://www.investopedia.com/terms/r/rsi.asp): Detects overbought and undersold conditions {float64}
  Simple Moving Average [(SMA)](https://www.investopedia.com/terms/s/sma.asp): The average closing price over a selected time period {float64}
  Corr: The correlation between the closing price (C) and the Simple Moving Average (SMA) {float64}
  Parabolic Stop And Reverse [(SAR)](https://www.investopedia.com/terms/p/parabolicindicator.asp): Detects trend direction {float64}
  Average Directional Index [(ADX)](https://www.investopedia.com/terms/w/wilders-dmi-adx.asp): Estimates trend strength {float64}
  Raw variables of T-1: The OCHL variables of the previous timepoint {float64}
  Opening difference (OO): Difference between the current open value (O) and the open value (O) at T-1 {float64}
  Open close difference (OC): Difference between the current open value (O) and the close value (C) at T-1 {float64}
  Ret: Percent gain in open price compared to the previous (T-1) time periods {float64}
  Return(X=1-10): Percent gain in open price at time point T-X {float64}
  
Rows with raw incomplete stock data were deleted as were rows with NaN or Null values that might arise during the calculation of indicators or derivatives.
---
MODEL
