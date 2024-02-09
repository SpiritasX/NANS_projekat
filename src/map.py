from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle


def map_of_serbia(ax):
    return Basemap(llcrnrlon=18.7,llcrnrlat=41.7,urcrnrlon=23.2,urcrnrlat=46.3, resolution='i', projection='merc', ax=ax)

def edit_map_of_serbia(m):
    m.drawcountries()
    m.drawcounties(color='b')
    m.drawcoastlines()
    m.fillcontinents()
    return m
   


def draw_stations(m, lons, lats):
    lon, lat = m(lons, lats)
    m.plot(lon, lat, 'co')
    plt.show()

#c = Circle(xy=m(19.721709, 45.208386), radius=(m.ymax - m.ymin) / 3, color='black')
#ax1.add_patch(c)

#x, y = m(19.721709, 45.208386)
#m.plot(x, y, 'co')
