import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import map

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


def save_clusters(df_spojeno, df):
    list_of_models = []
    for datum in df.index:
        temp = [[float(x) for x in list(df_spojeno.loc['Latitude'])], [float(x) for x in list(df_spojeno.loc['Longitude'])], [float(x[:-1]) for x in list(df_spojeno.loc['Nadmorska visina'])], list(df[df_spojeno.columns].loc[datum])]
        temp = np.array(temp).T.tolist()
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(temp)
        temp = np.array(temp).T.tolist()
        temp[3] = kmeans.labels_
        temp = np.array(temp).T.tolist()
        list_of_models.append(temp)
    return list_of_models


draw = None
marker_map = {0: 'green', 1: 'cornflowerblue', 2: 'orange', 3: 'red', 4:'purple'}
fig, ax = plt.subplots()


def generate_image(frame): # saljem 365 frejmova
    print(frame)
    print(str(frame / len(draw) * 100) + "%")

    ax.clear()

    m = map.map_of_serbia()
    m = map.edit_map_of_serbia(m)

    for lat, lon, vis, c in draw[frame]:
        lon, lat = m(lon, lat)
        m.plot(lon, lat, color=marker_map[c], marker='o', ax=ax)


def clusters_to_video(data):
    global draw
    draw = data
    # Postavljanje animacije
    zeljena_trajanja = 10
    zeljeni_fps = len(data) / zeljena_trajanja
    intervall = 1000 / zeljeni_fps

    #mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\tijan\\OneDrive\\Desktop\\ffmpeg-N-113561-ge05d3c1a16-win64-gpl\\bin\\ffmpeg.exe'
    ani = FuncAnimation(fig, generate_image, frames=len(data), interval=intervall)

    # Prikazivanje animacije
    #plt.show()
    #video_writer = animation.FFMpegWriter(fps=70)
    #ani.save('C:\\Users\\tijan\\OneDrive\\Desktop\\rkoanp\\animacija.gif', writer=video_writer)

    #GIF
    writergif = animation.PillowWriter(fps=zeljeni_fps)
    ani.save('data\\animacija.gif', writer=writergif)
    # Na osnovu grafika, odaberite optimalan broj klastera i ažurirajte n_clusters u sledećem koraku
