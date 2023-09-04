# Stock-Trend-Predictor
A web application that analyzes and predicts future stock trends of companies
The entire project has been done using python. IDEs used: Jupyter notebook, VSCode
Data Source: Yahoo finance data from 2012-2022
ML model used: LSTM

## Libraries used
Data analysis and predictions: pandas, pandas_datareader, numpy, matplotlib, yfinance, keras, tensorflow, datetime
Web application: streamlit

## How the web app works
All data predictions in this web app are restricted to the closing prices of the stock. Enter the stock ticker of the company whose stock you want to predict. It first shows the data visualizations of Closing price vs Time, Closing price vs Time with 100ma as well as with both 100ma and 200ma. Further, it also displays the visualizations of stock closing prices for the next 30 days from the current date.

NOTE: Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy.
