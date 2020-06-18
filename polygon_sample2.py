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
from polygon_sample_functions import *

CONFIG = 'Europe'
THRESHOLD = 0
POINT_COUNT = 16
FILENAME_LOWRES = "Countries_110/ne_110m_admin_0_countries.shp"
FILENAME_HIGHRES = "Countries_110/ne_110m_admin_0_countries.shp"
#FILENAME_HIGHRES = "Countries_50/ne_50m_admin_0_countries.shp"
#FILENAME_HIGHRES = "Admin_1_10/ne_10m_admin_1_states_provinces_lakes.shp"
#FILENAME_LOWRES = FILENAME_HIGHRES
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

def prune(shape, record, threshold):
    '''Remove small islands etc.'''
    triangulation = triangulation_cache[record[configs.configs[CONFIG]['name_identifier']]]
    new_shape = []
    for i in range(len(shape)):
        area = functions.polygon_area(triangulation[i])
        if area > threshold:
            new_shape.append(shape[i])

    return new_shape

def plot_shape_recs(shape_recs, color=True):
    if color:
        color_mapping = get_colors_from_shape_recs(shape_recs)

    for shape, record in shape_recs:
        for polygon in shape:
            poly = Polygon(polygon)
            x,y = poly.exterior.xy
            ax.plot(x, y, color='000', alpha=1,
                linewidth=1, zorder=0)
            if color:
                ax.fill(x, y, color=map_colors[record['MAPCOLOR13'] - 1], alpha=1,
                    linewidth=0, zorder=0)
            else:
                ax.fill(x, y, color=(1, 1, 1), alpha=1,
                    linewidth=0, zorder=0)

#plot_shape_recs(shape_read.shapefile_to_shape_recs(FILENAME_HIGHRES), color=False)

shape_recs = shape_read.shapefile_to_shape_recs(FILENAME_LOWRES)
shape_recs = prepare_shape_recs(shape_recs)
trimmed_recs = []

all_points = []
for shape, rec in shape_recs:
    for polygon in shape:
        for point in polygon:
            all_points.append(point)


print("Computing triangulation...")
for shape, rec in tqdm(shape_recs):
    if not any([f(rec) for f in configs.configs[CONFIG]['exclude']]):
        triangulation = []
        for polygon in shape:
            tri = functions.triangulate_polygon(polygon)
            triangulation.append(tri)
        triangulation_cache[rec[configs.configs[CONFIG]['name_identifier']]] = triangulation

shape_recs = [(prune(shape, record, THRESHOLD), record) for shape, record in shape_recs]
split_dict = {'a': []}
for shape, rec in shape_recs:
    if not any([f(rec) for f in configs.configs[CONFIG]['show_but_exclude']]):
        if configs.configs[CONFIG].get('grouping'):
            if configs.configs[CONFIG]['grouping'](rec) not in split_dict:
                split_dict[configs.configs[CONFIG]['grouping'](rec)] = [(shape, rec)]
            else:
                split_dict[configs.configs[CONFIG]['grouping'](rec)].append((shape, rec))
        else:
            split_dict['a'].append((shape, rec))
    continue

plot_shape_recs(prepare_shape_recs(shape_read.shapefile_to_shape_recs(FILENAME_HIGHRES)))
all_triangulation = []
for region in triangulation_cache:
    for shape in triangulation_cache[region]:
        for part in shape:
            all_triangulation.append(part)

if WATER:
    water_sample = functions.sample_points_in_water(all_triangulation, 100, *configs.configs[CONFIG]['trim_bounds'])

point_sets = []
circles = []
for region in split_dict:
    point_sets_local = []
    for shape, rec in split_dict[region]:
        sample = functions.sample_shape(shape, rec, POINT_COUNT, triangulation_cache[rec[configs.configs[CONFIG]['name_identifier']]], THRESHOLD)
        point_sets_local.append(sample)
        point_sets.append(sample)
    
    if WATER:
        point_sets_local.append(water_sample)
    
    discs = functions.smallest_k_disc_facade(point_sets_local, water_constraint=WATER_CONSTRAINT, region_constraint=REGION_CONSTRAINT) # How many regions in outer circle
    for disc in discs:
        circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)
        circles.append(circle)

if PLOT_WATER_POINTS:
    x, y = zip(*water_sample)
    ax.plot([x], [y], marker='o', markersize=3, c=(0, 0, 0), zorder=3)

if SHOW_TRIANGULATION:
    for polygon in all_triangulation:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=1, zorder=0)

if PLOT_POINTS:
    x, y = zip(*point_sets[0])
    ax.plot([x], [y], marker='o', markersize=3, c=(0, 0, 0), zorder=3)
    for i in range(1, len(point_sets)):
        x, y = zip(*point_sets[i])
        ax.plot([x], [y], marker='o', markersize=3, c=(0.3, 0.3, 0.3), zorder=3)

for c in circles:
    ax.add_artist(c)

plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
