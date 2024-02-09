import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import map
from functools import partial


marker_map = {0: 'green', 1: 'cornflowerblue', 2: 'orange', 3: 'red', 4:'purple'}


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


def save_clusters(result):
    list_of_models = []
    for datum in result:
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(datum)
        datum = np.array(datum).T.tolist()
        datum[1] = kmeans.labels_
        datum = np.array(datum).T.tolist()
        list_of_models.append(datum)
    return list_of_models


def generate_image(frame, data, m):
    print(frame)
    print(str(frame / len(data) * 100) + "%")

    for lat, lon, vis, c in data[frame]:
        lon, lat = m(lon, lat)
        m.plot(lon, lat, color=marker_map[c], marker='o')


def clusters_to_video(data):
    fig, ax = plt.subplots()
    m = map.map_of_serbia(ax)
    m = map.edit_map_of_serbia(m)

    # Postavljanje animacije
    zeljena_trajanja = 10
    zeljeni_fps = len(data) / zeljena_trajanja
    intervall = 1000 / zeljeni_fps

    #mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\tijan\\OneDrive\\Desktop\\ffmpeg-N-113561-ge05d3c1a16-win64-gpl\\bin\\ffmpeg.exe'
    ani = FuncAnimation(fig, partial(generate_image, data=data, m=m), frames=len(data), interval=intervall)

    #GIF
    writergif = animation.PillowWriter(fps=zeljeni_fps)
    ani.save('data\\animacija.gif', writer=writergif)
