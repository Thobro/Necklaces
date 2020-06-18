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
from shapely.ops import unary_union
from shapely.geometry import Point
import pyclipper
from shapely.geometry import Polygon
import sys

def shapefile_to_shape_recs(fp):
    sf = shapefile.Reader(fp)
    data = gpd.read_file(fp)
    data_proj = data.copy()
    data_proj = data_proj.to_crs(epsg=3395)
    #data_proj = data_proj.to_crs("+proj=robin +lon_0=0 +x_0=0 +y_0=0 +a=6371000 +b=6371000 +units=m +no_defs")
    print(sf.fields)

    shapes = []
    raw_shapes = []
    
    
    #print(data_proj['geometry'][122])
    for entry in data_proj['geometry']:
        if entry.geom_type == 'MultiPolygon':
            parts = []
            raw_shapes.append(entry)
            for polygon in entry:
                parts.append(list(polygon.exterior.coords))
            shapes.append(parts)
        elif entry.geom_type == 'Polygon':
            parts = []
            raw_shapes.append(entry)
            parts.append(list(entry.exterior.coords))
            shapes.append(parts)

    
    shape_recs_sf = sf.shapeRecords()
    shape_recs = [(parts, shape_rec.record) for (parts, shape_rec) in zip(shapes, shape_recs_sf)]
    #shape_recs_raw = [(parts, shape_rec.record) for (parts, shape_rec) in zip(raw_shapes, shape_recs_sf)]
    

    
    # Offset and save
    '''polygons = []
    for shape, rec in shape_recs:
        if rec['admin'] == 'Netherlands':
            for part in shape:
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(part, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
                solution = pco.Execute(100000.0)
                for polygon in solution:
                    p = Polygon(polygon)
                    polygons.append(p)

    boundary = gpd.GeoSeries(unary_union(polygons))
    with open("polygon.wkt", "w") as text_file:
        text_file.write(str(boundary[0]))'''
    

    return shape_recs

