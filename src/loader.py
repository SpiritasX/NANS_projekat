import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def load_all_tables(plot=False):
    dfs = []
    for year in range(2011, 2021):
        dfs.append(pd.read_csv(f'..\\data\\{year}\\{year}-co_max8h.csv'))

    for df in dfs:
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

    df = pd.concat(dfs).set_index('Datum').sort_values(by='Datum')

    for column in df.columns.difference(['Datum']):
        df[column] = pd.to_numeric(df[column], errors='coerce')

    if plot:
        df.plot(legend=None)
        plt.show()

    return df
