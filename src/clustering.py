from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from map import edit_map_of_serbia
#|from utils_nans1 import *

# Normalizacija podataka
def normalization(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    #print(data_scaled)
    return data_scaled  

# Pronalaženje optimalnog broja klastera pomoću metode "Elbow"
def elbowMethod(data_scaled):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    # Prikazivanje rezultata metode "Elbow"
    plt.plot(range(1, 11), wcss)
    plt.title('Metoda Elbow za određivanje optimalnog broja klastera')
    plt.xlabel('Broj klastera')
    plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
    plt.show()

# Transponovanje stanice.csv
def transposing(stanice): # tabela stanice.csv
    stanice_transposed = stanice.transpose().set_index(0)
    for i in range(1, len(stanice_transposed.columns) + 1):
        stanice_transposed = stanice_transposed.rename(columns={i: stanice_transposed[i]['Naziv stanice']})
    stanice_transposed = stanice_transposed[1:]
    #print(stanice_transposed)
    return stanice_transposed

# Spajamo dve tabele pomocu inner join-a
def innerJoinTables(stanice_transposed, data):
    df_spojeno = pd.concat([stanice_transposed, data], join='inner')
    df_spojeno.index.rename('Parametri', inplace=True)
    #print(df_spojeno)
    return df_spojeno

def clustering(df_spojeno, data, m):
    i = 0
    marker_map = {0: 'green', 1: 'cornflowerblue', 2: 'orange', 3: 'red', 4:'purple'}

    for datum in data.index:
        #datum = '2019-01-01'
        map =  edit_map_of_serbia(m)
        temp = [[float(x) for x in list(df_spojeno.loc['Latitude'])], [float(x) for x in list(df_spojeno.loc['Longitude'])], [float(x[:-1]) for x in list(df_spojeno.loc['Nadmorska visina'])], list(data[df_spojeno.columns].loc[datum])]
        temp = np.array(temp).T.tolist()
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(temp)
        temp = np.array(temp).T.tolist()
        temp[3] = kmeans.labels_
        temp = np.array(temp).T.tolist()
        for lat, lon, vis, c in temp:
            lon, lat = map(lon, lat)
            map.plot(lon, lat, color=marker_map[c], marker='o')
        
        if i == 8:
            break
        else:
            i+=1
            plt.show()

# animacija


