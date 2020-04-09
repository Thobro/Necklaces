from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read

fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111, aspect="equal")
ax.axis('off')

shape_recs = shape_read.shapefile_to_shape_recs()
shape_recs = [(shape, record) for shape, record in shape_recs if record['CONTINENT'] == "South America"]
for shape, record in shape_recs:
    if record['POP_EST'] < 10000:
        continue
    for polygon in shape:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=2, solid_capstyle='round', zorder=2)

point_sets = []
for shape, record in shape_recs:
    if record['POP_EST'] < 10000:
        continue
    sample = functions.sample_shape(shape, 10)
    point_sets.append(sample)

disc = functions.smallest_k_disc(point_sets)
circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3)

for i in range(len(point_sets)):
    x, y = zip(*point_sets[i])
    ax.plot([x], [y], marker='o', markersize=3, c=np.random.rand(3,))

ax.add_artist(circle)
plt.show()
