import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt


matplotlib.rcParams['figure.figsize'] = (8, 3)
sb.set(font_scale=1.)

# NO2C:data\linreg\NO2.csv
#zadatak 1
df = pd.read_csv('data\linreg\NO2.csv', sep = ',')
df.head()

df = df.dropna()

x = df.drop(columns=['Kotlarnica Nova fabrika Jagodina'])
y = df['Kotlarnica Nova fabrika Jagodina']
model = get_fitted_model(x, y)

df_test = pd.read_csv('data\linreg\testNO2.csv', sep=',')
x_test = df_test.drop(columns=['AirPollution'])
y_test = df_test['AirPollution']

test_rmse = get_rsquared(model, x_test, y_test)
print(f'rmse: {test_rmse}')
'''
#zadatak 2
min_expected_raise, max_expected_raise = get_conf_interval(model,'IndustrialEmissions', alpha=0.05)
print(f'min_expected_raise: {min_expected_raise: .2f}')
print(f'max_expected_raise: {max_expected_raise: .2f}')

autocorrelation, _ = independence_of_errors_assumption(model, sm.add_constant(x), y, plot=False)
if autocorrelation is None:
    print('vrednosti su validen jer je zadovoljena pretpostavka o nezavisnosti greske')
else:
    print('vrednosti nisu validne')

#zadatak 3
df = pd.read_csv('data/train.csv')
print(check_for_missing_values(df))

print(model.summary())

#ovde sa dmoram da nagadjam
df['TrafficDensity'] = df['TrafficDensity'].interpolate(method='linear', limit_direction='both')

#dropovanje
df = df.drop(columns=['WindSpeed', 'Temperature', 'GreenSpace', 'Wind'])

#perfect_collinearity_assumption(df, True)

x = df.drop(columns=['AirPollution'])
y = df['AirPollution']
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=True)

model = get_fitted_model(x_train, y_train)
print(are_assumptions_satisfied(model,x_train,y_train))

val_rmso = get_rmse(model, x_val, y_val)
print(f'validation rmse:{val_rmso: .2f}')

df_test = pd.read_csv('data/test.csv', sep=',')
x_test = df_test.drop(columns=['AirPollution','WindSpeed', 'Temperature', 'GreenSpace', 'Wind'])
y_test = df_test['AirPollution']
test_rmse = get_rmse(model,x_test,y_test)
print(f'test rmse: {test_rmse:.2f}')
'''