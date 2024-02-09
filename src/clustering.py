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
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(datum)
        datum = np.array(datum).T.tolist()
        datum[3] = kmeans.labels_
        datum = np.array(datum).T.tolist()
        list_of_models.append(datum)
    return list_of_models


def generate_image(frame, data, ax, m):
    print(frame)
    print(str(frame / len(data) * 100) + "%")

    ax.clear()
    m.drawcountries()
    m.drawcoastlines()

    for date in data[frame]:
        lon, lat = m(date[1], date[0])
        ax.plot(lon, lat, color=marker_map[date[3]], marker='o')

def clusters_to_video(data, num_of_frames_from_end):
    # Set up animation parameters
    desired_duration = 10
    desired_fps = num_of_frames_from_end / desired_duration
    interval = 1000 / desired_fps

    fig, ax = plt.subplots()
    m = map.map_of_serbia(ax)

    ani = FuncAnimation(fig, partial(generate_image, data=data[-num_of_frames_from_end:], ax=ax, m=m), frames=num_of_frames_from_end, interval=interval)

    writergif = animation.PillowWriter(fps=desired_fps)
    ani.save('data\\animacija.gif', writer=writergif)
