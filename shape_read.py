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

def shapefile_to_shape_recs(fp):
    #fp = "NE_110_CU/ne_110m_admin_0_map_units.shp"
    #fp = "Provinces/ne_110m_admin_1_states_provinces_lakes.shp"
    #fp = "Countries_50/ne_50m_admin_0_countries_lakes.shp"
    #fp = "Countries_50/ne_50m_admin_0_countries.shp"
    #fp = "Countries_110/ne_110m_admin_0_countries.shp"
    
    sf = shapefile.Reader(fp)
    data = gpd.read_file(fp)
    data_proj = data.copy()
    data_proj = data_proj.to_crs(epsg=3395)
    #data_proj = data_proj.to_crs("+proj=robin +lon_0=0 +x_0=0 +y_0=0 +a=6371000 +b=6371000 +units=m +no_defs")
    print(sf.fields)

    shapes = []
    #print(data_proj['geometry'][122])
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
    #shape_recs = [shape_recs[112]]
    #2.18300*10**6, 2.18500*10**6, 4.11500*10**6, 4.11400*10**6
    '''for idx, (shape, rec) in enumerate(shape_recs):
        for p in shape:
            for q in p:
                x, y = q
                if 2.18300*10**6 <= x <= 2.18500*10**6 and -4.11500*10**6 <= y <= -4.11400*10**6:
                    print(idx)'''

    return shape_recs

