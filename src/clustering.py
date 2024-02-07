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
def innerJoinTables(dfs):
    df_spojeno = pd.concat(dfs, join='inner')
    #df_spojeno.index.rename('Parametri', inplace=True)
    #print(df_spojeno)
    return df_spojeno

# Obicno iscrtavanje frejmova
def clustering(df_spojeno, data, m): 
    #i = 0
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
        
        # if i == 8:
        #     break
        # else:
        #     i+=1
        #     plt.show()

# ANIMACIJA
'''
# Inicijalizacija praznog grafa
fig, ax = plt.subplots()

i = 0
# Funkcija koja generiše slike
def generate_image(frame): # saljem 365 frejmova

    ax.clear()
    datum = data.index[frame] #pristupam trazenom frejmu
    
    # for datum in data.index:
    #     datum = '2019-01-01'
    m.drawcountries()
    m.drawcounties(color='b')
    m.drawcoastlines()
    m.fillcontinents()

    temp = [[float(x) for x in list(df_spojeno.loc['Latitude'])], [float(x) for x in list(df_spojeno.loc['Longitude'])], [float(x[:-1]) for x in list(df_spojeno.loc['Nadmorska visina'])], list(data[df_spojeno.columns].loc[datum])]
    temp = np.array(temp).T.tolist()

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(temp)
    temp = np.array(temp).T.tolist()
    temp[3] = kmeans.labels_
    temp = np.array(temp).T.tolist()

    for lat, lon, vis, c in temp:
        lon, lat = m(lon, lat)
        m.plot(lon, lat, color=marker_map[c], marker='o')

    # if i == 4:
    #     break
    # else:
    #     i+=1
    #     plt.show()
        
# Postavljanje animacije
zeljena_trajanja = 10
zeljeni_fps = len(data.index) / zeljena_trajanja
intervall = 1000 / zeljeni_fps

#mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\tijan\\OneDrive\\Desktop\\ffmpeg-N-113561-ge05d3c1a16-win64-gpl\\bin\\ffmpeg.exe'
ani = FuncAnimation(fig, generate_image, frames=len(data.index), interval=intervall)

# Prikazivanje animacije
#plt.show()     
#video_writer = animation.FFMpegWriter(fps=70)
#ani.save('C:\\Users\\tijan\\OneDrive\\Desktop\\rkoanp\\animacija.gif', writer=video_writer)  

#GIF
writergif = animation.PillowWriter(fps=zeljeni_fps) 
ani.save('C:\\Users\\tijan\\OneDrive\\Desktop\\rkoanp\\animacija.gif', writer=writergif)
# Na osnovu grafika, odaberite optimalan broj klastera i ažurirajte n_clusters u sledećem koraku
optimalni_broj_klastera = 3 # postavite optimalan broj klastera
'''