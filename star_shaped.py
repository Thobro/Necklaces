from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read
from map_colors import map_colors
import random
from tqdm import tqdm
import configs

CONFIG = 'Europe'
THRESHOLD = 0
POINT_COUNT = 12
FILENAME_LOWRES = "Physical_110/ne_110m_land.shp"
PLOT_POINTS = False
SHOW_TRIANGULATION = False
WATER = False
PLOT_WATER_POINTS = False
WATER_CONSTRAINT = 100
REGION_CONSTRAINT = 100

triangulation_cache = {}

fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

def trim(shape, trim_bounds):
    x_min, x_max, y_min, y_max = trim_bounds
    for i in range(len(shape)):
        shape[i] = [v for v in shape[i] if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max]

    return [p for p in shape if len(p) != 0]
    

def prune(shape, record, threshold):
    '''Remove small islands etc.'''
    triangulation = triangulation_cache[record[configs.configs[CONFIG]['name_identifier']]]
    new_shape = []
    for i in range(len(shape)):
        area = functions.polygon_area(triangulation[i])
        if area > threshold:
            new_shape.append(shape[i])

    return new_shape

def prepare_shape_recs(shape_recs):
    shape_recs = [(trim(shape, configs.configs[CONFIG]['trim_bounds']), record) for shape, record in shape_recs if any([f(record) for f in configs.configs[CONFIG]['options']]) and all([f(record) for f in configs.configs[CONFIG]['requirements']])]
    shape_recs = [(shape, record) for shape, record in shape_recs if len(shape) != 0]
    shape_recs = [(shape, record) for shape, record in shape_recs if not any([f(record) for f in configs.configs[CONFIG]['exclude']])]
    return shape_recs

def get_colors_from_shape_recs(shape_recs):
    c = 0
    color_mapping = {}

    for shape1, rec1 in shape_recs:
        neighbors = [r[configs.configs[CONFIG]['name_identifier']] for (s, r) in shape_recs if r[configs.configs[CONFIG]['name_identifier']] in color_mapping and s != shape1 and functions.borders(s, shape1)]
        neighbor_colors = [color_mapping[n] for n in neighbors if n in color_mapping]
        while c in neighbor_colors:
            c += 1
        color_mapping[rec1[configs.configs[CONFIG]['name_identifier']]] = c % len(map_colors)

        c += 1
        c = c % len(map_colors)
    
    return color_mapping

def plot_shape_recs(shape_recs, color=True):
    for shape, record in shape_recs:
        for polygon in shape:
            poly = Polygon(polygon)
            x,y = poly.exterior.xy
            ax.plot(x, y, color='000', alpha=1,
                linewidth=1, zorder=0)
            if color:
                ax.fill(x, y, color=map_colors[0], alpha=1,
                    linewidth=0, zorder=0)
            else:
                ax.fill(x, y, color=(1, 1, 1), alpha=1,
                    linewidth=0, zorder=0)

shape_recs = shape_read.shapefile_to_shape_recs(FILENAME_LOWRES)
#shape_recs = prepare_shape_recs(shape_recs)
trimmed_recs = []

all_points = []
for shape, rec in shape_recs:
    for polygon in shape:
        for point in polygon:
            all_points.append(point)


print("Computing triangulation...")
'''for shape, rec in tqdm(shape_recs):
    triangulation = []
    for polygon in shape:
        tri = functions.triangulate_polygon(polygon)
        triangulation.append(tri)'''

split_dict = {'a': []}
for shape, rec in shape_recs:
    split_dict['a'].append((shape, rec))

plot_shape_recs(shape_read.shapefile_to_shape_recs(FILENAME_LOWRES))

#x, y = zip(*p)
#ax.plot([x], [y], marker='o', markersize=3, c=(1, 1, 1), zorder=3)

plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
