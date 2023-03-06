import datetime
import math

import numpy as np
import pandas as pd
import writeinflux

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


def co231(client):
    train_data = pd.read_csv('CO_231.csv')

    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_231': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]  # Час пояса убираю

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data['Concentration'] < 5)]  # указывем выбросы

    measurement = f"co231"
    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": measurement,
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def co231_pr(client):
    train_data_CO2 = pd.read_csv('CO_231.csv')
    train_data_CO2.rename(columns={'Unnamed: 0': 'Date', '74040126_231': 'Concentration'}, inplace=True)
    train_data_CO2['Date'] = train_data_CO2['Date'].str[:-6]

    train_data_CO2['Date'] = pd.to_datetime(train_data_CO2['Date'])
    train_data_CO2.set_index('Date', inplace=True)

    train_data_CO2 = train_data_CO2[(train_data_CO2['Concentration'] < 5)]  # указывем выбросы

    df_CO2 = train_data_CO2[['Concentration']]

    forecast_out = int(math.ceil(0.05 * len(df_CO2)))
    print(forecast_out)
    df_CO2['label'] = train_data_CO2['Concentration'].shift(-forecast_out)

    scaler = StandardScaler()
    X = np.array(df_CO2.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)

    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained

    df_CO2.dropna(inplace=True)
    y = np.array(df_CO2['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test, y_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)

    rg = Ridge()
    rg.fit(X_train, y_train)
    rg_confidence = rg.score(X_test, y_test)

    names = ['Linear Regression', 'Random Forest', 'Ridge']
    columns = ['Date', 'Concentration']
    scores = [lr_confidence, rf_confidence, rg_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    last_date = df_CO2.index[-1]  # получение последней даты в наборе данных
    last_unix = last_date.timestamp()  # преобразование его во время в секундах
    one_day = 86400  # one day = 86400 seconds
    next_unix = last_unix + one_day  # получение времени в секундах на следующий день
    forecast_set = rf.predict(X_Predictions)  # прогнозирование прогнозных данных
    df_CO2['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_CO2.loc[next_date] = [np.nan for _ in range(len(df_CO2.columns) - 1)] + [i]



    measurement = f"co231pr"
    field_name = f"Concentration"

    for index, row in df_CO2.iterrows():
        time = index
        concentration = row['Forecast']

        if math.isnan(concentration):
            continue

        json_body1 = [
            {
                "measurement": measurement,
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h239(client):
    train_data = pd.read_csv('H_239.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_239': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 80) & (train_data.Concentration > 50)]  # указывем выбросы

    measurement = f"h239"
    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": measurement,
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h239_pr(client):
    train_data_H239 = pd.read_csv('H_239.csv')
    train_data_H239.rename(columns={'Unnamed: 0': 'Date', '74040126_239': 'Concentration'}, inplace=True)
    train_data_H239['Date'] = train_data_H239['Date'].str[:-6]

    train_data_H239['Date'] = pd.to_datetime(train_data_H239['Date'])
    train_data_H239.set_index('Date', inplace=True)

    train_data_H239.isnull().sum()#Проверка нулевых значений в наборе данных

    train_data_H239 = train_data_H239[(train_data_H239.Concentration < 80) & (train_data_H239.Concentration > 50)]

    df_H239 = train_data_H239[['Concentration']]

    forecast_out = int(math.ceil(0.05 * len(df_H239)))
    print(forecast_out)
    df_H239['label'] = train_data_H239['Concentration'].shift(-forecast_out)

    scaler = StandardScaler()
    X = np.array(df_H239.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)

    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained

    df_H239.dropna(inplace=True)
    y = np.array(df_H239['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test, y_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)

    rg = Ridge()
    rg.fit(X_train, y_train)
    rg_confidence = rg.score(X_test, y_test)

    names = ['Linear Regression', 'Random Forest', 'Ridge']
    columns = ['Date', 'Concentration']
    scores = [lr_confidence, rf_confidence, rg_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    last_date_H239 = df_H239.index[-1]  # получение последней даты в наборе данных
    last_unix = last_date_H239.timestamp()  # преобразование его во время в секундах
    one_day = 86400  # one day = 86400 seconds
    next_unix = last_unix + one_day  # получение времени в секундах на следующий день
    forecast_set = rf.predict(X_Predictions)  # прогнозирование прогнозных данных
    df_H239['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_H239.loc[next_date] = [np.nan for _ in range(len(df_H239.columns) - 1)] + [i]

    measurement = f"h239pr"
    field_name = f"Concentration"

    for index, row in df_H239.iterrows():
        time = index
        concentration = row['Forecast']

        if math.isnan(concentration):
            continue

        json_body1 = [
            {
                "measurement": measurement,
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h2s234(client):
    train_data = pd.read_csv('H2S_234.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_234': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration > 0.00004)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'h2s234',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h2s234_pr(client):
    train_data_H2S_234 = pd.read_csv('H2S_234.csv')
    train_data_H2S_234.rename(columns={'Unnamed: 0': 'Date', '74040126_234': 'Concentration'}, inplace=True)
    train_data_H2S_234['Date'] = train_data_H2S_234['Date'].str[:-6]

    train_data_H2S_234['Date'] = pd.to_datetime(train_data_H2S_234['Date'])
    train_data_H2S_234.set_index('Date', inplace=True)

    train_data_H2S_234 = train_data_H2S_234[(train_data_H2S_234.Concentration > 0.00004)]
    df_H2S_234 = train_data_H2S_234[['Concentration']]

    forecast_out = int(math.ceil(0.05 * len(df_H2S_234)))
    print(forecast_out)
    df_H2S_234['label'] = train_data_H2S_234['Concentration'].shift(-forecast_out)

    scaler = StandardScaler()
    X = np.array(df_H2S_234.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)

    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained

    df_H2S_234.dropna(inplace=True)
    y = np.array(df_H2S_234['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test, y_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)

    rg = Ridge()
    rg.fit(X_train, y_train)
    rg_confidence = rg.score(X_test, y_test)

    names = ['Linear Regression', 'Random Forest', 'Ridge']
    columns = ['Date', 'Concentration']
    scores = [lr_confidence, rf_confidence, rg_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    last_date = df_H2S_234.index[-1]  # получение последней даты в наборе данных
    last_unix = last_date.timestamp()  # преобразование его во время в секундах
    one_day = 86400  # one day = 86400 seconds
    next_unix = last_unix + one_day  # получение времени в секундах на следующий день
    forecast_set = rf.predict(X_Predictions)  # прогнозирование прогнозных данных
    df_H2S_234['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_H2S_234.loc[next_date] = [np.nan for _ in range(len(df_H2S_234.columns) - 1)] + [i]

    field_name = f"Concentration"

    for index, row in df_H2S_234.iterrows():
        time = index
        concentration = row['Forecast']

        if math.isnan(concentration):
            continue

        json_body1 = [
            {
                "measurement": "h2s234pr",
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h240(client):
    train_data = pd.read_csv('H_240.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_240': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration > 735 )&(train_data.Concentration < 744)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'h240',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def h240_pr(client):
    train_data_H_240 = pd.read_csv('H_240.csv')
    train_data_H_240.rename(columns={'Unnamed: 0': 'Date', '74040126_240': 'Concentration'}, inplace=True)
    train_data_H_240['Date'] = train_data_H_240['Date'].str[:-6]

    train_data_H_240['Date'] = pd.to_datetime(train_data_H_240['Date'])
    train_data_H_240.set_index('Date', inplace=True)

    train_data_H_240 = train_data_H_240[(train_data_H_240.Concentration > 735) & (train_data_H_240.Concentration < 744)]

    df_H_240 = train_data_H_240[['Concentration']]

    forecast_out = int(math.ceil(0.05 * len(df_H_240)))
    print("forecast_out = ", forecast_out)
    df_H_240['label'] = train_data_H_240['Concentration'].shift(-forecast_out)

    scaler = StandardScaler()
    X = np.array(df_H_240.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)

    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained

    df_H_240.dropna(inplace=True)
    y = np.array(df_H_240['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test, y_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)

    rg = Ridge()
    rg.fit(X_train, y_train)
    rg_confidence = rg.score(X_test, y_test)

    names = ['Linear Regression', 'Random Forest', 'Ridge']
    columns = ['Date', 'Concentration']
    scores = [lr_confidence, rf_confidence, rg_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    last_date = df_H_240.index[-1]  # получение последней даты в наборе данных
    last_unix = last_date.timestamp()  # преобразование его во время в секундах
    one_day = 86400  # one day = 86400 seconds
    next_unix = last_unix + one_day  # получение времени в секундах на следующий день
    forecast_set = rf.predict(X_Predictions)  # прогнозирование прогнозных данных
    df_H_240['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_H_240.loc[next_date] = [np.nan for _ in range(len(df_H_240.columns) - 1)] + [i]

    field_name = f"Concentration"

    for index, row in df_H_240.iterrows():
        time = index
        concentration = row['Forecast']

        if math.isnan(concentration):
            continue

        json_body1 = [
            {
                "measurement": "h240pr",
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def no2232(client):
    train_data = pd.read_csv('NO2_232.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_232': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 0.18)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'no2232',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def o3235(client):
    train_data = pd.read_csv('O3_235.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_235': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 0.03)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'o3235',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def pm10236(client):
    train_data = pd.read_csv('PM10_236.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_236': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 0.27)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'pm10236',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def pm25237(client):
    train_data = pd.read_csv('PM25_237.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_237': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 0.24)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'pm25237',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def so2233(client):
    train_data = pd.read_csv('SO2_233.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_233': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration < 0.074)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 'so2233',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)


def t238(client):
    train_data = pd.read_csv('T_238.csv')
    train_data.rename(columns={'Unnamed: 0': 'Date', '74040126_238': 'Concentration'}, inplace=True)
    train_data['Date'] = train_data['Date'].str[:-6]

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)

    train_data.isnull().sum()  # Проверка нулевых значений в наборе данных

    train_data = train_data[(train_data.Concentration > -19) & (train_data.Concentration < 30)]  # указывем выбросы

    field_name = f"Concentration"

    for index, row in train_data.iterrows():
        time = index
        concentration = row['Concentration']
        json_body1 = [
            {
                "measurement": 't238',
                "time": time,
                "fields": {
                    field_name: concentration
                }
            }
        ]

        insert = writeinflux.WriteInflux(json_body1)
        insert.export_to_influxdb(client)
