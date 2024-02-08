import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt
from loader import *
from utils import *


def making_table(location_name):
    # spajam tabele o zagadjenju za 2019. i 2020. godinu
    df = load_all_tables(years = [2019, 2020], file = 'pm2.5')
    for stanica in df:
        df = fillna_mean(df, stanica) 
    #print(df)

    location = pd.read_csv(f'data\LinReg\{location_name}.csv')

    location.insert(0, 'pm2.5', list(df[location_name]))
    location = location.set_index('datetime')
    #print(location)
    return location


def linear_regression(location):
        
    x = location.drop(columns=['pm2.5'])
    y = location['pm2.5']
    model = get_fitted_model(x, y)
    print(model.summary())
    
    location = location.drop(columns=['sealevelpressure', 'windspeed'])
    x = location.drop(columns=['pm2.5'])
    y = location['pm2.5']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=True)
    model = get_fitted_model(x_train, y_train)
    print(model.summary())
    print(are_assumptions_satisfied(model,x_train,y_train))

    test_rmse = get_rmse(model,x_test,y_test)
    print(f'test rmse: {test_rmse:.2f}')
    
    # ax = plt.figure().add_subplot(projection='3d')

    # x = np.linspace(0, 250, 731)
    # y = 78.2366 - 2.0604 * np.array(list(location['temp'])) - 0.3218 * np.array(list(location['humidity']))
    # ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

    # colors = ('r', 'g', 'b', 'k')
    
    # ax.scatter(location['temp'], location['humidity'], zs=location['pm2.5'])

    # ax.legend()
    # ax.set_xlabel('temp')
    # ax.set_ylabel('humidity')
    # ax.set_zlabel('pm2.5')

    # plt.show()

