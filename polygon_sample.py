from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np

fig = plt.figure(1, dpi=72)
ax = fig.add_subplot(111, aspect="equal")
ax.axis('off')
fig.patch.set_visible(False)

def triangle_area(t):
    A, B, C = t[0], t[1], t[2]
    return abs(0.5 * (((B[0] - A[0]) * (C[1] - A[1])) - ((C[0] - A[0]) * (B[1] - A[1]))))

def point_from_triangle(t):
    A, B, C = t[0], t[1], t[2]
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    p = ( A[0] + x * (B[0] - A[0]) + y * (C[0] - A[0]), A[1] + x * (B[1] - A[1]) + y * (C[1] - A[1]) )

    area = triangle_area(t)
    a1 = triangle_area([A, B, p])
    a2 = triangle_area([A, C, p])
    a3 = triangle_area([B, C, p])
    print(area, a1 + a2 + a3)
    if int(a1 + a2 + a3) == int(area):
        return p
    else:    
        return ( A[0] + A[0] + (B[0] - A[0]) + (C[0] - A[0]) - p[0], A[1] + A[1] + (B[1] - A[1]) + (C[1] - A[1]) - p[1] )

def triangulate_monotone(ps):
    polygons = []
    ps = sorted(ps, key=lambda x: x[1], reverse=True) # Sort on decreasing y
    u1 = ps[0]
    stack = [ps[0], ps[1]]
    for i in range(2, len(ps)):
        if ps[i][0] > u1[0] and stack[-1][0] < u1[0] or ps[i][0] < u1[0] and stack[-1][0] > u1[0]: # Hacky version of 'different chain'
            polygons.append([ps[i], stack.pop(), stack.pop()])
            stack.append(ps[i-1])
            stack.append(ps[i])
        

    return polygons

p1 = [
    (285.294, 691.677),
    (242.454, 656.397),
    (267.654, 595.917),
    (320.574, 626.157),
    (365.934, 605.997),
    (378.534, 633.717),
    (403.734, 651.357),
    (370.974, 674.037),
    (355.854, 699.237),
    (285.294, 691.677),
]

p2 = [(267.654, 595.917), (320.574, 626.157), (242.454, 656.397),]
p3 = [(365.934, 605.997), (320.574, 626.157), (378.534, 633.717),]
p4 = [(355.854, 699.237), (285.294, 691.677), (370.974, 674.037),]
p5 = [(285.294, 691.677), (242.454, 656.397), (370.974, 674.037),]
p6 = [(370.974, 674.037), (403.734, 651.357), (242.454, 656.397),]
p7 = [(378.534, 633.717), (403.734, 651.357), (242.454, 656.397),]
p8 = [(320.574, 626.157), (378.534, 633.717), (242.454, 656.397),]

triangulation = [p2, p3, p4, p5, p6, p7, p8]
#triangulation = [p2]
polygon_size = sum([triangle_area(t) for t in triangulation])
triangulation = [(t, triangle_area(t) / polygon_size) for t in triangulation]

triangulation[0] = triangulation[0][0], triangulation[0][1], 0
for i in range(1, len(triangulation)):
    triangulation[i] = (triangulation[i][0], triangulation[i][1] + triangulation[i-1][1], triangulation[i-1][1])

samples = np.random.uniform(0, 1, 200)
points = []
for s in samples:
    for t, ma, mi in triangulation:
        if mi <= s <= ma:
            p = point_from_triangle(t)
            points.append(p)
        

'''for polygon in [p2, p3, p4, p5, p6, p7, p8]:
    poly = Polygon(polygon)
    x,y = poly.exterior.xy
    ax.plot(x, y, color='000', alpha=1,
        linewidth=2, solid_capstyle='round', zorder=2)'''



poly = Polygon(p1)
#random_points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000)) for k in range(12)]
#poly = Polygon(random_points)
x,y = poly.exterior.xy
ax.plot(x, y, color='000', alpha=1,
    linewidth=2, solid_capstyle='round', zorder=2)

x, y = zip(*points)
ax.plot([x], [y], marker='o', markersize=3, color="blue")
plt.savefig("filename.pdf", bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

