from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import tripy
import math
from tqdm import tqdm
import shapefile
import functions
import geopandas as gpd

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
    fp = "NE_110_CU/ne_110m_admin_0_map_units.shp"
    sf = shapefile.Reader(fp)
    data = gpd.read_file(fp)
    data_proj = data.copy()
    #data_proj = data_proj.to_crs(epsg=3395)
    data_proj = data_proj.to_crs("+proj=robin +lon_0=0 +x_0=0 +y_0=0 +a=6371000 +b=6371000 +units=m +no_defs ")

    shapes = []

    for entry in data_proj['geometry']:
        if entry.geom_type == 'MultiPolygon':
            parts = []
            for polygon in entry:
                parts.append(list(polygon.exterior.coords))
            shapes.append(parts)
        elif entry.geom_type == 'Polygon':
            parts = []
            parts.append(list(entry.exterior.coords))
            shapes.append(parts)
    
    shape_recs = sf.shapeRecords()
    shape_recs = [(parts, shape_rec.record) for (parts, shape_rec) in zip(shapes, shape_recs)]

    #shape_recs = [(functions.shape_to_parts(shape_rec.shape), shape_rec.record) for shape_rec in shape_recs]
    return shape_recs


def project():
    fp = "NE_110_CU/ne_110m_admin_0_map_units"
    data = gpd.read_file(fp + '.shp')
    data_proj = data.copy()
    data_proj = data_proj.to_crs(epsg=3035)
    data_proj.to_file(f"{fp}_3035.shp")

    print(data_proj['geometry'].head())



#sf = shapefile.Reader("NE_110_CU/ne_110m_admin_0_map_units.shp")
#sf = shapefile.Reader("Provinces/ne_110m_admin_1_states_provinces_lakes.shp")
#fields = sf.fields
#shapeRecs = sf.shapeRecords()
#print(shapeRecs[100].record['CONTINENT'])

