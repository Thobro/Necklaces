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
shape_recs = [(shape, record) for shape, record in shape_recs if record['SUBREGION'] == "Western Europe" or record['SUBREGION'] == "Northern Europe" and record['POP_EST'] >= 1000]
#shape_recs = [(shape, record) for shape, record in shape_recs if record['CONTINENT'] == "Asia" and record['POP_EST'] >= 10000]
#shape_recs = [(shape, record) for shape, record in shape_recs if record['iso_a2'] == "NL"]
for shape, record in shape_recs:
    for polygon in shape:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=2, solid_capstyle='round', zorder=2)

point_sets = []
for shape, record in shape_recs:
    sample = functions.sample_shape(shape, 10)
    point_sets.append(sample)

disc = functions.smallest_k_disc(point_sets)
circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)
ax.add_artist(circle)

for i in range(len(point_sets)):
    x, y = zip(*point_sets[i])
    ax.plot([x], [y], marker='o', markersize=3, c=np.random.rand(3,))
plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
