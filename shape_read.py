from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import tripy
import math
from tqdm import tqdm
import shapefile
import functions

'''fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111, aspect="equal")
ax.axis('off')



#poly = Polygon(max(polygons, key=lambda x: len(x)))
#x,y = poly.exterior.xy
#ax.plot(x, y, color='000', alpha=1,
    #linewidth=2, solid_capstyle='round', zorder=2)


for polygon in polygons:
    poly = Polygon(polygon)
    x,y = poly.exterior.xy
    ax.plot(x, y, color='000', alpha=1,
        linewidth=2, solid_capstyle='round', zorder=2)


plt.show()'''

def shapefile_to_shape_recs():
    sf = shapefile.Reader("NE_110_CU/ne_110m_admin_0_map_units.shp")
    #sf = shapefile.Reader("Provinces/ne_110m_admin_1_states_provinces_lakes.shp")
    shape_recs = sf.shapeRecords()
    shape_recs = [(functions.shape_to_parts(shape_rec.shape), shape_rec.record) for shape_rec in shape_recs]
    '''for shape, rec in shape_recs:
        if rec['SUBREGION'] == 'Northern Europe':
            print(rec['NAME'])'''
    return shape_recs

#sf = shapefile.Reader("NE_110_CU/ne_110m_admin_0_map_units.shp")
sf = shapefile.Reader("Provinces/ne_110m_admin_1_states_provinces_lakes.shp")
fields = sf.fields
print(fields)
#shapeRecs = sf.shapeRecords()
#print(shapeRecs[100].record['CONTINENT'])

