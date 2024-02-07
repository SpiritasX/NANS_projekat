import pandas as pd


def fillna_mean(df, col_name, chunk = 40):
    '''Fill NA values with windowed mean.'''
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