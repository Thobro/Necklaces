from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read
from map_colors import map_colors
import random

fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

options = [
    lambda r: r['SUBREGION'] == "Northern Europe",
    lambda r: r['SUBREGION'] == "Western Europe",
    lambda r: r['SUBREGION'] == "Southern Europe",
    lambda r: r['SUBREGION'] == "Eastern Europe"
]

requirements = [
    lambda r: r['POP_EST'] >= 1000
]

trim_bounds = {
    'Europe': (-6*10**6, 4.2*10**6, -6*10**6, 7.4*10**6),
}

def trim(shape, key):
    if key not in trim_bounds:
        return shape
    x_min, x_max, y_min, y_max = trim_bounds[key]
    for i in range(len(shape)):
        shape[i] = [v for v in shape[i] if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max]

    shape = [p for p in shape if len(p) != 0]
    return [p for p in shape if len(p) != 0]

shape_recs = shape_read.shapefile_to_shape_recs()
shape_recs = [(trim(shape, "Europe"), record) for shape, record in shape_recs if any([f(record) for f in options]) and all([f(record) for f in requirements])]
shape_recs = [(shape, record) for shape, record in shape_recs if len(shape) != 0]
split_dict = {'a': []}
for shape, rec in shape_recs:
    split_dict['a'].append(shape)
    #if rec['NAME'] == "France":
    #    split_dict['a'].append(shape)
    continue
    if rec['region'] not in split_dict:
        split_dict[rec['region']] = [shape]
    else:
        split_dict[rec['region']].append(shape)

c = 0

for shape, record in shape_recs:
    c += 1
    c = c % len(map_colors)
    for polygon in shape:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=1, zorder=0)
        ax.fill(x, y, color=map_colors[c], alpha=1,
            linewidth=0, zorder=0)

point_sets = []
'''for shape, record in shape_recs:
    sample = functions.sample_shape(shape, 30)
    point_sets.append(sample)

disc = functions.smallest_k_disc_fast(point_sets)
circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)'''


circles = []
for region in split_dict:
    point_sets_local = []
    for shape in split_dict[region]:
        sample = functions.sample_shape(shape, 10)
        point_sets_local.append(sample)
        point_sets.append(sample)
    disc = functions.smallest_k_disc_fast(point_sets_local)
    circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)
    circles.append(circle)

'''for i in range(len(point_sets)):
    x, y = zip(*point_sets[i])
    ax.plot([x], [y], marker='o', markersize=3, c=random.choice(map_colors), zorder=0)'''

for c in circles:
    ax.add_artist(c)

plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
