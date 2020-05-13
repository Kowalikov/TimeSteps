import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

aapl = yf.Ticker("AAPL")
df = aapl.history(period="max")
for col in df.columns:
    print(col)
print("pierszy rząd", df.iloc[50])
#df = pd.read_csv(r'/Users/Marek/PycharmProjects/TimeSteps/data/AAPL.csv')
#print(df.info)
for col in df.columns:
    print(col)


#changing the data object to datetime typef
con=df['Name']
df['Name']=pd.to_datetime(df['Name'])
df.set_index('Name', inplace=True)
print(df.index)
print(df.sample(5, random_state=0))
"""
#add 3 columns with indexes of years, months and days
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
print(df.sample(5, random_state=0))

temp=df.groupby(['Date'])['Close'].mean()
temp.plot(figsize=(15,5), title= 'Closing Prices(Monthwise)', fontsize=14)
plt.show()

df.groupby('month')['Close'].mean().plot.bar()
plt.show()

test = df[250:]
train = df[:249]


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label ='Original')
    plt.plot(rolmean, color='red', label ='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    print("Results of dickey fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics',
                              'p - value',
                              'No.of lags used',
                              'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value( % s)' % key] = values
    print(output)


test_stationarity(train['Close'])

"""