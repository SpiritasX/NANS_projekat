import pandas as pd
from urllib.request import urlopen

zagadjivaci = ['pm2.5']#, 'pm10']#['co', 'no2', 'o3', 'pm2.5', 'pm10', 'so2']
godine = ['2019', '2020']

dfs = []
for godina in godine:
    for zagadjivac in zagadjivaci:
        dfs.append(pd.read_csv(f'data\\{godina}\\{godina}-{zagadjivac}.csv'))

df = pd.concat(dfs, join='inner')

stations = pd.read_csv('data\\stanice.csv').set_index('Naziv stanice')
# print(stations['Naziv stanice'][df.columns])
names = list(df.columns.drop('Datum'))
lats = list(stations['Latitude'][df.columns.drop('Datum')])
lons = list(stations['Longitude'][df.columns.drop('Datum')])

key = input('Input your API key: ')

for i in range(len(names)):
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lats[i]}%2C{lons[i]}/2019-01-01/2020-12-31?unitGroup=metric&elements=datetime%2Ctemp%2Chumidity%2Cwindspeed%2Cpressure&include=days&key={key}&contentType=csv'
    response = urlopen(url)
    with open("data\\LinReg\\" + names[i] + ".csv", "w") as fp:
        fp.write(response.read().decode())