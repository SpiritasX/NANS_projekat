import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def fillna_mean(df, col_name, chunk = 40):
    df_copy = df.copy()
    for i in range(0, len(df)):
        if not pd.isna(df_copy.loc[df.index[i], col_name]):
            continue
        a = int(max(0, i - chunk/2))
        b = int(min(len(df), i + chunk/2))
        chunk_values = df_copy.iloc[a:b][col_name]
        mean = chunk_values.mean()
        df_copy.loc[df.index[i], col_name] = mean
    return df_copy
    # df_copy = df.copy()
    # window = df_copy[col_name].rolling(window=chunk, min_periods=1, center=True)
    # df_copy[col_name] = df_copy[col_name].fillna(window.mean())
    # return df_copy


def load_all_tables(plot=False, years=[x for x in range(2011, 2021)], file='co'):
    dfs = []
    for year in years:
        try:
            dfs.append(pd.read_csv(f'data\\{year}\\{year}-{file}.csv'))
        finally:
            continue

    for df in dfs:
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

    df = inner_join_tables(dfs).set_index('Datum').sort_values(by='Datum')

    for column in df.columns.difference(['Datum']):
        df[column] = pd.to_numeric(df[column], errors='coerce')

    if plot:
        df.plot(legend=None)
        plt.show()

    return df


def load_stations():
    return pd.read_csv('data\\stanice.csv')


def normalize(data):
    return StandardScaler().fit_transform(data)


def transposing(stanice):
    stanice_transposed = stanice.transpose().set_index(0)
    for i in range(1, len(stanice_transposed.columns) + 1):
        stanice_transposed = stanice_transposed.rename(columns={i: stanice_transposed[i]['Naziv stanice']})
    stanice_transposed = stanice_transposed[1:]
    return stanice_transposed


def inner_join_tables(dfs):
    return pd.concat(dfs, join='inner')