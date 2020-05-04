from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read
from map_colors import map_colors
import random
import configs

CONFIG = 'Europe'

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

shape_recs = shape_read.shapefile_to_shape_recs()
shape_recs = [(trim(shape, configs.configs[CONFIG]['trim_bounds']), record) for shape, record in shape_recs if any([f(record) for f in configs.configs[CONFIG]['options']]) and all([f(record) for f in configs.configs[CONFIG]['requirements']])]
shape_recs = [(shape, record) for shape, record in shape_recs if len(shape) != 0]
shape_recs = [(shape, record) for shape, record in shape_recs if not any([f(record) for f in configs.configs[CONFIG]['exclude']])]
split_dict = {'a': []}
for shape, rec in shape_recs:
    if not any([f(rec) for f in configs.configs[CONFIG]['show_but_exclude']]):
        split_dict['a'].append(shape)
    continue
    if rec['region'] not in split_dict:
        split_dict[rec['region']] = [shape]
    else:
        split_dict[rec['region']].append(shape)



c = 0
color_mapping = {}

for shape1, rec1 in shape_recs:
    neighbors = [r['NAME'] for (s, r) in shape_recs if r['NAME'] in color_mapping and s != shape1 and functions.borders(s, shape1)]
    neighbor_colors = [color_mapping[n] for n in neighbors if n in color_mapping]
    while c in neighbor_colors:
        c += 1
    color_mapping[rec1['NAME']] = c % len(map_colors)

    c += 1
    c = c % len(map_colors)


for shape, record in shape_recs:
    for polygon in shape:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=1, zorder=0)
        ax.fill(x, y, color=map_colors[color_mapping[record['NAME']]], alpha=1,
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
        sample = functions.sample_shape(shape, 16)
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
