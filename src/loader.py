import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def load_all_tables(plot=False, years=[x for x in range(2011, 2021)], file='co'):
    dfs = []
    for year in years:
        try:
            dfs.append(pd.read_csv(f'data\\{year}\\{year}-{file}.csv'))
        finally:
            continue

    for df in dfs:
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

    df = pd.concat(dfs, join = 'inner').set_index('Datum').sort_values(by='Datum')

    for column in df.columns.difference(['Datum']):
        df[column] = pd.to_numeric(df[column], errors='coerce')

    if plot:
        df.plot(legend=None)
        plt.show()

    return df


def load_stations():
    return pd.read_csv('data\\stanice.csv')