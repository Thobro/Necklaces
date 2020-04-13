from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read

fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

shape_recs = shape_read.shapefile_to_shape_recs()
#shape_recs = [(shape, record) for shape, record in shape_recs if record['SUBREGION'] == "Northern America" or record['SUBREGION'] == "Central America" and record['POP_EST'] >= 1000]
#shape_recs = [(shape, record) for shape, record in shape_recs if record['CONTINENT'] == "Asia" and record['POP_EST'] >= 10000]
shape_recs = [(shape, record) for shape, record in shape_recs if record['name'] not in ['Alaska', 'Hawaii']]
split_dict = {}
for shape, rec in shape_recs:
    if rec['region'] not in split_dict:
        split_dict[rec['region']] = [shape]
    else:
        split_dict[rec['region']].append(shape)


for shape, record in shape_recs:
    for polygon in shape:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=2, solid_capstyle='round', zorder=2)

point_sets = []
'''for shape, record in shape_recs:
    sample = functions.sample_shape(shape, 30)
    point_sets.append(sample)'''

circles = []
for region in split_dict:
    point_sets_local = []
    for shape in split_dict[region]:
        sample = functions.sample_shape(shape, 30)
        point_sets_local.append(sample)
        point_sets.append(sample)
    disc = functions.smallest_k_disc_fast(point_sets_local)
    circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)
    circles.append(circle)

for c in circles:
    ax.add_artist(c)

for i in range(len(point_sets)):
    x, y = zip(*point_sets[i])
    ax.plot([x], [y], marker='o', markersize=3, c=np.random.rand(3,))
plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
