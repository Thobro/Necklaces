import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
import functions
import math
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from geomdl import multi
import xmltodict

fig = plt.figure(1, figsize=(7,7))
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

geo = []
with open("africa_clean.ipe") as file:
    data = file.read()
    data = xmltodict.parse(data)
    pts = data['ipe']['page']['path']['#text'].split('\n')
    for p in pts:
        x, y, c = p.split(' ')
        geo.append((float(x), float(y)))

print(geo)


def get_points_on_tangent(p, a, dist):
    dx = math.cos(a) * dist
    dy = math.sin(a) * dist
    return (p[0] + dx, p[1] + dy), (p[0] - dx, p[1] - dy)


center = (220, 550)
cm = multi.CurveContainer()
cm.vis = VisMPL.VisCurve2D(axes=False, labels=False,
                           ctrlpts=False, legend=False)
pts = [[485, 475], [430, 366], [325, 231], [274, 438], [
    193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]

pts = [[274, 540], [252, 483], [209, 441], [188, 525], [154, 525], [136, 560], [158, 593], [191, 596], [233, 582]]

radius = max([functions.euclid_dist(p, center) for p in pts])
#pts = [functions.project_to_circle(p, center, radius) for p in pts]

tangents = []
for i in range(len(pts)):
    p1 = pts[(i-1) % len(pts)]
    p2 = pts[(i+1) % len(pts)]
    pr = (p2[0] - p1[0], p2[1] - p1[1])
    alpha = math.atan2(pr[1], pr[0])
    tangents.append(alpha)

ctrlpts = []
for i in range(len(pts)):
    c = BSpline.Curve()
    c.degree = 3
    dist = functions.euclid_dist(pts[i], pts[(i+1) % len(pts)])
    d = dist * 0.25
    p1, p2 = get_points_on_tangent(pts[i], tangents[i], d)
    if math.atan2(p1[1] - center[1], p1[0] - center[0]) < math.atan2(p2[1] - center[1], p2[0] - center[0]):
        p1, p2 = p2, p1

    p3, p4 = get_points_on_tangent(pts[(i+1) % len(pts)], tangents[(i+1) % len(pts)], d)
    if math.atan2(p3[1] - center[1], p3[0] - center[0]) < math.atan2(p4[1] - center[1], p4[0] - center[0]):
        p3, p4 = p4, p3

    c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
    c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    cm.add(c)


for c in cm:
    curvepts = np.array(c.evalpts)
    ctrlpts = np.array(c.ctrlpts)
    #cppolygon, = plt.plot(ctrlpts[:, 0], ctrlpts[:, 1], color='black', linestyle='-.', marker='o', markersize='3')
    curveplt, = plt.plot(curvepts[:, 0], curvepts[:, 1], color='green', linestyle='-')  # evaluated curve points

poly = Polygon(geo)
x, y = poly.exterior.xy
ax.plot(x, y, color='000', alpha=1,
        linewidth=1, zorder=0)
plt.show()

#cm.render()


