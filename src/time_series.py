import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def walk_forward_loop(train_df, val_df, column_name, order=(0, 0, 0)):
    history = train_df[column_name].copy() # trening skup koji prosirujemo stvarnom vrednosti
    wf_pred = pd.Series() # serija predikcija koju iterativno popunjavamo

    for i in range(len(val_df)):
        wf_model = ARIMA(history, order=order).fit()
        # sacuvaj predikciju
        y_pred = wf_model.forecast(steps=1)
        wf_pred = pd.concat([wf_pred, y_pred])
        # sacuvaj stvarnu vrednost u trening skup
        true_value = pd.Series(data=val_df.iloc[i][column_name], index=[val_df.index[i]])
        history = pd.concat([history, true_value])

    return wf_pred


def analyse_time_series(df):
    df = fillna_mean(df, 'Kikinda Centar', chunk=40)

    df['Kikinda Centar'].plot()
    plt.show()

    plot_acf(df['Kikinda Centar'], lags=30)
    plt.show()
    plot_pacf(df['Kikinda Centar'], lags=30)
    plt.show()

    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    p, d, q = 17, 0, 4
    wf_pred = walk_forward_loop(train, test, 'Kikinda Centar', (p, d, q))
    plt.plot(df['Kikinda Centar'])
    plt.plot(wf_pred)
    plt.show()

    # arima_model = ARIMA(train['Kikinda Centar'], order=(p, d, q)).fit()
    # y_pred_arima = arima_model.predict(start=train.index[p+1], end=df.index[-1])
    #
    # plt.plot(train['Kikinda Centar'])
    # plt.plot(test['Kikinda Centar'], color='orange')
    # plt.plot(y_pred_arima, color='green')
    # plt.show()

    # stl = STL(df['Kikinda Centar']).fit()
    # trend, seasonal, resid = stl.trend, stl.seasonal, stl.resid
    # stl.plot()
    # plt.show()

    # reconstructed_original_data = trend + seasonal + resid
    #
    # plt.plot(reconstructed_original_data, '.', label='rekonstruisani podaci')
    # plt.plot(df['Kikinda Centar'], '--', label='stvarni podaci')
    # plt.legend()
    # plt.show()


def time_series_trend(df, col_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col_name], color='blue')
    plt.title('Air Pollution Data Over Time')
    plt.xlabel('Date')
    plt.ylabel('Pollution Level')
    plt.grid(True)
    plt.show()

    rolling_mean = df[col_name].rolling(window=30, min_periods=1).mean()  # Assuming a 30-day moving average
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col_name], color='blue', label='Original')
    plt.plot(rolling_mean.index, rolling_mean, color='red', label='30-Day Rolling Mean')
    plt.title('Air Pollution Data with 30-Day Rolling Mean')
    plt.xlabel('Date')
    plt.ylabel('Pollution Level')
    plt.legend()
    plt.grid(True)
    plt.show()

    # yearly_trend = air_pollution_data.resample('Y').mean()
    # plt.figure(figsize=(10, 6))
    # plt.plot(yearly_trend.index.year, yearly_trend['Kikinda Centar'], marker='o', color='red')
    # plt.title('Yearly Trend in Pollution Levels')
    # plt.xlabel('Year')
    # plt.ylabel('Mean Pollution Level')
    # plt.grid(True)
    # plt.show()


def decompose(df, col_name, plot=True):
    result = seasonal_decompose(df[col_name], model='additive', period=365)
    if plot:
        result.plot()
        plt.show()
    return result


def yearly(df, col_name):
    result = decompose(df, col_name, False)
    seasonal_subseries = df[col_name] - result.trend - result.resid
    seasonal_subseries = seasonal_subseries.groupby(seasonal_subseries.index.month).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(seasonal_subseries.index, seasonal_subseries, marker='o', color='green')
    plt.title('Seasonal Subseries Plot')
    plt.xlabel('Month')
    plt.ylabel('Mean Pollution Level')
    plt.grid(True)
    plt.show()


def weekly(df, col_name):
    weekly_trend = df.groupby(df.index.dayofweek)[col_name].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_trend.index, weekly_trend, marker='o', color='green')
    plt.title('Weekly Trend in Pollution Levels')
    plt.xlabel('Day of the Week')
    plt.ylabel('Mean Pollution Level')
    plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.grid(True)
    plt.show()