import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sktime as sk
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.arima import AutoARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sktime.forecasting.compose import make_reduction
from sktime.transformations.series.detrend import Detrender,Deseasonalizer
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import MeanRelativeAbsoluteError
from sktime.performance_metrics.forecasting import MeanSquaredScaledError
from sktime.performance_metrics.forecasting import GeometricMeanAbsoluteError
from codes import *
rmsse = MeanSquaredScaledError(square_root=True)
gmae = GeometricMeanAbsoluteError()


def smape(actual, forecast):
    return (1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast)))).iloc[0]

def reshape_func(x):
    x1 = np.reshape(x,(x.shape[0],x.shape[1],-1)) # first merge the sensor with the feature
    x2 = np.reshape(x1,(-1,x1.shape[2]))  # then merge the hours with 5 min intervals
    return x2

def fillzeros(x):
    x = x.replace(0,np.nan)
    x = x.fillna(method="ffill")
    return x

def model(train,test,forecaster,detrender):
    fh = ForecastingHorizon(test.index, is_relative=False)
    if detrender == True:
        detrender = Detrender()
        ytrain_det = detrender.fit_transform(train)
        forecaster = ThetaForecaster(deseasonalize = False)
        forecaster.fit(ytrain_det)
        pred = forecaster.predict(fh)
        pred = detrender.inverse_transform(pred)
        smape1 = smape(test,pred)
        rmsse1 = rmsse(test,pred,y_train=train)
        gmae1 = gmae(test,pred)
    else:
        forecaster.fit(train)
        pred = forecaster.predict(fh)
        smape1 = smape(test,pred)
        rmsse1 = rmsse(test,pred,y_train=train)
        gmae1 = gmae(test,pred)
    return [smape1,rmsse1,gmae1,pred]


def hourly_mean(train,test,N):
    h_train = []
    h_test = []
    for i in range(N):
        h_train.append(np.mean(train[0][i*(12):(i+1)*12-1]))
        h_test.append(np.mean(test[0][i*(12):(i+1)*12-1]))
    return [pd.DataFrame(h_train),pd.DataFrame(h_test)]

def daily_mean(train,test,N):
    d_train = []
    d_test = []
    for i in range(N):
        d_train.append(np.mean(train[0][i*(12)*24:(i+1)*12*24-1]))
        d_test.append(np.mean(test[0][i*(12)*24:(i+1)*12*24-1]))
    return [pd.DataFrame(d_train),pd.DataFrame(d_test)]
        
def weekly_mean(train,test,N):
    w_train = []
    w_test = []
    for i in range(N):
        w_train.append(np.mean(train[0][i*(12)*24*7:(i+1)*12*24*7-1]))
        w_test.append(np.mean(test[0][i*(12)*24*7:(i+1)*12*24*7-1]))
    return [pd.DataFrame(w_train),pd.DataFrame(w_test)]
        