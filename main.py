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
from pandas.core.frame import DataFrame as dataframe
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def loadStockData(name="AAPL", monthMean=False):
    #reading raw stock data and historical extraction for data frame
    rawStockData = yf.Ticker("AAPL")
    df = rawStockData.history(period="max")

    #show samples
    print(df.index)
    print(df.sample(5, random_state=0))

    #add 3 columns with indexes of years, months and days
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['date'] = df.index.date
    df['index'] = list(range(df.index.size))
    print(df.sample(5, random_state=0))

    #show stocks plot
    temp=df.groupby(['Date'])['Close'].mean()
    temp.plot(figsize=(15,5), title= 'Closing Prices(Monthwise)', fontsize=14)
    plt.show()

    #optional month mean for analysis
    if monthMean==True:
        df.groupby('month')['Close'].mean().plot.bar()
        plt.show()
    #returning data frame with Close, Open prices with row labels ad date, and additional date columns (Year, Day, Month)
    return df

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

def timePeriodIndex(actionsDF, timePeriod):
    startTrain = actionsDF.loc[actionsDF['date']==timePeriod]
    i=1
    while startTrain.__sizeof__() ==0:
        startDate= datetime.date(timePeriod.year, timePeriod.month, timePeriod.day+i)
        startTrain = actionsDF.loc[actionsDF['date'] == startDate]
        i+=1

    return startTrain['index'][0]


actionsDF=loadStockData(name="AAPL")

#split data for training and testing
timePeriods = (datetime.date(2000, 1, 1), datetime.date(2019, 1, 2), datetime.date(2019, 1, 2), datetime.date(2020, 1, 2)) #time range for training and for testing
timePeriodsIndexes = []
for x in range(4):
    timePeriodsIndexes.append(timePeriodIndex(actionsDF, timePeriods[x]))

train = actionsDF[timePeriodsIndexes[0]:timePeriodsIndexes[1]]
test = actionsDF[timePeriodsIndexes[2]:timePeriodsIndexes[3]]



#stationary test: mean and standard deviation
test_stationarity(train['Close'])

#deleting the trend procedure
train_log = np.log(train['Close'])
test_log = np.log(test['Close'])
moving_avg = train_log.rolling(24).mean()
plt.plot(train_log)
plt.plot(moving_avg, color = 'red')
plt.show()
train_log_moving_avg_diff = train_log - moving_avg
#droping empty values used in mean calc
train_log_moving_avg_diff.dropna(inplace = True), test_stationarity(train_log_moving_avg_diff)
#differating for stablilising the values
train_log_diff = train_log - train_log.shift(1)
test_stationarity(train_log_diff.dropna())
#now the time series is decomposed into trend and seasonality (periodic fluctuations)


#make the ARIMA model for the prediction
model = auto_arima(train_log, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train_log)
#make the prediction
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test_log.index,columns=['Prediction'])
#plot the predictions for validation set
plt.plot(train_log, label='Train')
plt.plot(test_log, label='Test')
plt.plot(forecast, label='Prediction')
plt.title('APPLE Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#judge the results by RMSE
rms = math.sqrt(mean_squared_error(test_log,forecast))
print("RMSE: ", rms)
