import pandas as pd
import pandas_datareader as data
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta

st.title('Stock Trend Predictor')

st.write('Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. ')

st.subheader('Stock Tickers for some companies as mentioned in yahoo finance')
tickers={'COMPANY': ['Microsoft', 'Waalmart', 'Apple', 'SBI', 'Google', 'Tesla', 'TCS', 'Reliance', 'Tata motors', 'Infosys'],
         'TICKERS': ['MSFT', 'WMT', 'AAPL', 'SBIN.NS', 'GOOG', 'TSLA', 'TCS', 'RELIANCE.BO', 'TATAMOTORS.BO', 'INFY']}
tickers_df=pd.DataFrame(tickers)
html_table = tickers_df.to_html(index=False, escape=False)
st.write(html_table, unsafe_allow_html=True)
st.write('\n\n')

start='2010-01-01'
end='2019-12-31'

st.subheader('Enter stock ticker')
user_input=st.text_input('', 'AAPL')
df=yf.download(user_input, start=start, end=end)

st.subheader('Data from 2012-2022')
st.write("Data from 2012 to 2022 has been used to train the machine learning model. Here's the data summary.")
st.write(df.describe())

st.subheader('Closing price vs Time')
fig=plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Closing price')
plt.plot(df.Close, label='Closing price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing price vs Time with 100MA')
fig1=plt.figure(figsize=(12, 6))
ma100=df.Close.rolling(100).mean()
plt.xlabel('Time')
plt.ylabel('Closing price')
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, color='r', label='MA100')
plt.legend()
st.pyplot(fig1)

st.subheader('Closing price vs Time with 100MA and 200MA')
fig2=plt.figure(figsize=(12, 6))
ma200=df.Close.rolling(200).mean()
plt.xlabel('Time')
plt.ylabel('Closing price')
plt.plot(ma100, color='r', label='MA100')
plt.plot(ma200, color='g', label='MA200')
plt.plot(df.Close, label='Closing price')
plt.legend()
st.pyplot(fig2)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0, 1))
data_training_array=scaler.fit_transform(data_training)

model=load_model('keras_model.keras')

past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Original vs Prediction')
fig3=plt.figure(figsize=(12, 6))
plt.plot(y_test, color='b', label='Original Price')
plt.plot(y_predicted, color='r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
st.pyplot(fig3)

st.subheader('Closing Price vs Time for the next 30 days')
start_date=(datetime.now()-timedelta(days=100)).strftime('%Y-%m-%d')
end_date=datetime.now().strftime('%Y-%m-%d')
new_data=yf.download(user_input, start=start_date, end=end_date)
new_data=new_data.reset_index()
new_data=new_data.drop(['Date', 'Adj Close'], axis=1)

new_scaler=MinMaxScaler(feature_range=(0, 1))
new_data_training_array=new_scaler.fit_transform(pd.DataFrame(new_data['Close']))

x_test_new=[new_data_training_array[-100:]]
x_test_new=np.array(x_test_new)

new_scaler=new_scaler.scale_
scale_factor_new=1/new_scaler[0]
predicted_prices=[]

# Predict next 30 days
for i in range(30):
    y_predicted_new=model.predict(x_test_new)
    y_predicted_new=y_predicted_new*scale_factor_new
    
    # Append the prediction and remove the oldest data point
    x_test_new=np.append(x_test_new, y_predicted_new[-1])  
    x_test_new=x_test_new[1:]
    x_test_new = x_test_new.reshape(1, -1, 1)

    desired_sequence_length = 100

    if x_test_new.shape[1] < desired_sequence_length:
        padding_width = desired_sequence_length - x_test_new.shape[1]
        zero_padding = np.zeros((x_test_new.shape[0], padding_width, x_test_new.shape[2]))
        x_test_new = np.concatenate((zero_padding, x_test_new), axis=1)
    elif x_test_new.shape[1] > desired_sequence_length:
    # If the sequence is longer, you can truncate it to the desired length
        x_test_new = x_test_new[:, :desired_sequence_length, :]

    predicted_prices.append(y_predicted_new[-1])

date_rng=pd.date_range(start=end_date, periods=30, freq='D')
predicted_df=pd.DataFrame({'Date':date_rng, 'Predicted Price':predicted_prices})

date_values = predicted_df['Date'].values
predicted_price_values = predicted_df['Predicted Price'].values

fig4=plt.figure(figsize=(12, 6))
plt.plot(date_values, predicted_price_values, color='r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.xticks(rotation='vertical')
plt.legend()
st.pyplot(fig4)