# Project1

Data
---
Originates from the Yahhoo Finance API

For a selected timeperiod the following 23 variables are available at every selected time interval:

**RAW (used as is)**

- Open (O): The opening price of the stock {float}
- Close (C): The closing price of the stock {float}
- High (H): The highest price of the stock in the interval {float} 
- Low (L): The lowest price of the stock in the interval {float}

**INDICATORS AND DERIVATIVES (Calculated using the raw variables)**

- Revative Strength Index [(RSI)](https://www.investopedia.com/terms/r/rsi.asp): Detects overbought and undersold conditions {float}
- Simple Moving Average [(SMA)](https://www.investopedia.com/terms/s/sma.asp): The average closing price over a selected time period {float}
- Corr: The correlation between the closing price (C) and the Simple Moving Average (SMA) {float64}
- Parabolic Stop And Reverse [(SAR)](https://www.investopedia.com/terms/p/parabolicindicator.asp): Detects trend direction {float}
- Average Directional Index [(ADX)](https://www.investopedia.com/terms/w/wilders-dmi-adx.asp): Estimates trend strength {float}
- Raw variables of T-1: The OCHL variables of the previous timepoint {float64}
- Opening difference (OO): Difference between the current open price (O) and the open price (O) at T-1 {float}
- Open close difference (OC): Difference between the current open price (O) and the close price (C) at T-1 {float}
- Ret: Percent gain in open price compared to the previous (T-1) time periods {float}
- Return(X=1-10): Percent gain in open price at time point T-X {float}
  
*Rows with raw incomplete stock data were deleted as were rows with NA or NaN values that might arise during the calculation of indicators or derivatives.*

Model
---
By default the Support Vector Machine (SVM) model is trained on 80% of the data and tested on the other 20%.
Three groups are created based on the Ret variable: the lowest 34% Ret observations are considerd as Sell and get assigned a -1 Signal, the highest 33% Ret observations are classified as Buys and get assigned a 1 Signal, all other orbservations are classified as hold and get assigned a 0 Signal. The Signal is stored in the Signal variable {int}.

The C and gamma parameters of the SVM are automatically optimised using a randomised search and an Radial Basis Function (RBF) is used as kernel. 

