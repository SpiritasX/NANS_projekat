import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.linear_model import LinearRegression


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


def load_all_tables(years, file, plot=False):
    if years is None:
        years = [x for x in range(2011, 2021)]
    if file is None:
        file = 'co'

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
        df.plot()#legend=None)
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


def calculate_residuals(model, features, labels):
    '''Calculates residuals between true value `labels` and predicted value.'''
    y_pred = model.predict(features)
    df_results = pd.DataFrame({'Actual': labels, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    return df_results


def linear_assumption(model: LinearRegression| RegressionResultsWrapper, features: np.ndarray|pd.DataFrame, labels: pd.Series, p_value_thresh=0.05, plot=True):
    '''
    Linear assumption: assumees linear relation between the independent and dependent variables to be linear.
    Testing linearity using the F-test.

    Interpretation of `p-value`:
    - `p-value >= p_value_thresh` indicates linearity between.
    - `p-value < p_value_thresh` doesn't indicate linearity.

    Returns (only if the model is from `statsmodels` not from `scikit-learn`):
    - is_linearity_found: A boolean indicating whether the linearity assumption is supported by the data.
    - p_value: The p-value obtained from the linearity test.
    '''
    df_results = calculate_residuals(model, features, labels)
    y_pred = df_results['Predicted']

    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(labels, y_pred, alpha=.5)
        # x = y line
        line_coords = np.linspace(np.concatenate([labels, y_pred]).min(), np.concatenate([labels, y_pred]).max())
        plt.plot(line_coords, line_coords, color='darkorange', linestyle='--')
        plt.title('Linear assumption')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

    if type(model) == RegressionResultsWrapper:
        p_value = model.f_pvalue
        is_linearity_found = True if p_value < p_value_thresh else False
        return is_linearity_found, p_value
    else:
        pass


def independence_of_errors_assumption(model, features, labels, plot=True):
    '''
    Independence of errors: assumes independent errors. 
    Assumes that there is no autocorrelation in the residuals. 
    Testing autocorrelation using Durbin-Watson Test.
    
    Interpretation of `d` value:
    - 1.5 <= d <= 2: No autocorrelation (independent residuals).
    - d < 1.5: Positive autocorrelation.
    - d > 2: Negative autocorrelation.

    Returns:
    - autocorrelation: The type of autocorrelation ('positive', 'negative', or None).
    - dw_value: The Durbin-Watson statistic value.
    '''
    df_results = calculate_residuals(model, features, labels)

    if plot:
        sb.scatterplot(x='Predicted', y='Residuals', data=df_results)
        plt.axhline(y=0, color='darkorange', linestyle='--')
        plt.show()

    from statsmodels.stats.stattools import durbin_watson
    dw_value = durbin_watson(df_results['Residuals'])

    autocorrelation = None
    if dw_value < 1.5: autocorrelation = 'positive'
    elif dw_value > 2: autocorrelation = 'negative'
    else: autocorrelation = None
    return autocorrelation, dw_value


def normality_of_errors_assumption(model, features, label, p_value_thresh=0.05, plot=True):
    '''
    Normality of errors: assumes normally distributed residuals around zero.
    Testing using the Anderson-Darling test for normal distribution on residuals.
    Interpretation of `p-value`:
    - `p-value >= p_value_thresh` indicates normal distribution.
    - `p-value < p_value_thresh` indicates non-normal distribution.

    Returns:
    - dist_type: A string indicating the distribution type ('normal' or 'non-normal').
    - p_value: The p-value from the Anderson-Darling test.
    '''
    df_results = calculate_residuals(model, features, label)
    
    if plot:
        plt.title('Distribution of residuals')
        sb.histplot(df_results['Residuals'], kde=True, kde_kws={'cut':3})
        plt.show()

    from statsmodels.stats.diagnostic import normal_ad
    p_value = normal_ad(df_results['Residuals'])[1]
    dist_type = 'normal' if p_value >= p_value_thresh else 'non-normal'
    return dist_type, p_value


def equal_variance_assumption(model, features, labels, p_value_thresh=0.05, plot=True):
    '''
    Equal variance: assumes that residuals have equal variance across the regression line.
    Testing equal variance using Goldfeld-Quandt test.
    
    Interpretation of `p-value`:
    - `p-value >= p_value_thresh` indicates equal variance.
    - `p-value < p_value_thresh` indicates non-equal variance.

    Returns:
    - dist_type: A string indicating the distribution type ('eqal' or 'non-eqal').
    - p_value: The p-value from the Goldfeld-Quandt test.
    '''
    df_results = calculate_residuals(model, features, labels)

    if plot:
        sb.scatterplot(x='Predicted', y='Residuals', data=df_results)
        plt.axhline(y=0, color='darkorange', linestyle='--')
        plt.show()

    if type(model) == LinearRegression:
        features = sm.add_constant(features)
    p_value =  sm.stats.het_goldfeldquandt(df_results['Residuals'], features)[1]
    dist_type = 'equal' if p_value >= p_value_thresh else 'non-equal'
    return dist_type, p_value


def perfect_collinearity_assumption(features: pd.DataFrame, plot=True):
    '''
    Perfect collinearity: assumes no perfect correlation between two or more features.
    Testing perfect collinearity between exactly two features using correlation matrix.

    Returns:
    - `has_perfect_collinearity`: A boolean indicating if perfect collinearity was found.
    '''
    correlation_matrix = features.corr() # racunamo matricu korelacije

    if plot:
        sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
        plt.title('Matrica korelacije')
        plt.show()
    
    np.fill_diagonal(correlation_matrix.values, np.nan)
    pos_perfect_collinearity = (correlation_matrix > 0.999).any().any()
    neg_perfect_collinearity = (correlation_matrix < -0.999).any().any()
    has_perfect_collinearity = pos_perfect_collinearity or neg_perfect_collinearity
    return has_perfect_collinearity


def are_assumptions_satisfied(model, features, labels, p_value_thresh=0.05):
    '''Check if all assumptions in multiple linear regression are satisfied.'''
    x_with_const = sm.add_constant(features)
    is_linearity_found, p_value = linear_assumption(model, x_with_const, labels, p_value_thresh, plot=False)
    autocorrelation, dw_value = independence_of_errors_assumption(model, x_with_const, labels, plot=False)
    n_dist_type, p_value = normality_of_errors_assumption(model, x_with_const, labels, p_value_thresh, plot=False)
    e_dist_type, p_value = equal_variance_assumption(model, x_with_const, labels, p_value_thresh, plot=False)
    has_perfect_collinearity = perfect_collinearity_assumption(features, plot=False)

    if is_linearity_found and autocorrelation is None and n_dist_type == 'normal' and e_dist_type == 'equal' and not has_perfect_collinearity:
        return True
    else:
        return False



def get_fitted_model(features, labels):
    '''Fits the model usting `statsmodels` package.'''
    x_with_const = sm.add_constant(features, has_constant='add')
    model = sm.OLS(labels, x_with_const).fit()
    return model


def check_for_missing_values(df): 
    '''Find missing values and return non-zero missing value counts.'''
    missing_values = df.isna().sum()
    non_zero_missing = missing_values[missing_values != 0]
    non_zero_missing_percentage = (non_zero_missing / len(df)) * 100
    return pd.DataFrame({
        'N missing': non_zero_missing,
        '% missing': non_zero_missing_percentage
    })


def visualize_column(df, col_name,  df_fixed=None):
    '''plot a single column values in a specified column.'''
    x = df.index
    plt.plot(x, df[col_name], 'bo', alpha=.2, label='original')
    if df_fixed is not None: plt.plot(x, df_fixed[col_name], 'r-', label='fixed')
    plt.legend()
    plt.show()


def get_sse(model, features, labels):
    '''Returns SSE score.'''
    y_pred = model.predict(sm.add_constant(features, has_constant='add'))
    sse = np.sum(((labels - y_pred) ** 2))
    return sse


def get_rmse(model, features, labels):
    '''Returns RMSE score.'''
    y_pred = model.predict(sm.add_constant(features, has_constant='add'))
    rmse = np.sqrt(np.mean(((labels - y_pred) ** 2)))
    return rmse


def get_rsquared(model, features, labels):
    '''Returns r^2 score.'''
    y_pred = model.predict(sm.add_constant(features, has_constant='add'))

    from sklearn.metrics import r2_score
    r_squared = r2_score(labels, y_pred)
    return r_squared

def get_rsquared_adj(model, features, labels):
    '''Returns adjusted r^2 score.'''
    num_attributes = features.shape[1]
    y_pred = model.predict(sm.add_constant(features, has_constant='add'))

    from sklearn.metrics import r2_score
    r_squared = r2_score(labels, y_pred)
    n = len(y_pred)
    p = num_attributes
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return adjusted_r_squared


def get_conf_interval(model: RegressionResultsWrapper, feature_name:str, alpha=0.05):
    '''
    Calculate a confidence interval for a specific feature's coefficient in the model.

    Parameters:
    model (RegressionResultsWrapper): The fitted linear regression model.
    feature_name (str): The name of the feature for which you want to calculate the confidence interval.
    alpha (float, optional): The desired significance level (default is 0.05 for a 95% confidence interval).

    Returns:
    Tuple (min_value, max_value): The lower and upper bounds of the confidence interval for the specified feature's coefficient.
    '''
    min_value, max_value = model.conf_int(alpha).loc[feature_name]
    return min_value, max_value

def get_pred_interval(model, features: int | float | np.ndarray | pd.Series | pd.DataFrame, p_value_trash=0.05):
    '''
    Calculate prediction interval for the given features.
    '''
    if type(features) == int or type(features) == float:
        pred_intervals = model.get_prediction(sm.add_constant([features, 0])).summary_frame(alpha=p_value_trash)
        low = pred_intervals['obs_ci_lower'].values[0]
        high = pred_intervals['obs_ci_upper'].values[0]
        return low, high
    
    if type(features) == list or type(features) == np.ndarray:
        const = np.array([1])
        datapoint = np.concatenate([features, const])
        pred_intervals = model.get_prediction(datapoint).summary_frame(alpha=p_value_trash)
        low = pred_intervals['obs_ci_lower'].values[0]
        high = pred_intervals['obs_ci_upper'].values[0]
        return low, high

    else:
        pred_intervals = model.get_prediction(sm.add_constant(features)).summary_frame(alpha=p_value_trash)
        low = pred_intervals['obs_ci_lower'].values
        high = pred_intervals['obs_ci_upper'].values
        return low, high
    

def get_rsquared_adj(y_true, y_pred, num_attributes):
    from sklearn.metrics import r2_score
    r_squared = r2_score(y_true, y_pred)
    n = len(y_pred)
    p = num_attributes
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return adjusted_r_squared

def get_fitted_model(x, y):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    return model

def fit_and_get_rsquared_adj_test(x_train, x_test, y_train, y_test):
    '''pomoćna funkcija koja vraca fitovan model i prilagodjeni r^2 nad test skupom.'''
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()
    y_pred = model.predict(sm.add_constant(x_test))
    adj_r2 = get_rsquared_adj(y_true=y_test, y_pred=y_pred, num_attributes=x_test.shape[1])
    return model, adj_r2
