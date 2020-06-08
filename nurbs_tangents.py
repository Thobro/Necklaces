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

INTERPOLATION_FACTOR = 0.5

geo = []
with open("africa_clean.ipe") as file:
    data = file.read()
    data = xmltodict.parse(data)
    pts = data['ipe']['page']['path']['#text'].split('\n')
    for p in pts:
        x, y, c = p.split(' ')
        geo.append((float(x), float(y)))

def get_points_on_tangent(p, a, dist):
    dx = math.cos(a) * dist
    dy = math.sin(a) * dist
    return (p[0] + dx, p[1] + dy), (p[0] - dx, p[1] - dy)



#center = (196, 520)
pts = [[274, 540], [252, 483], [209, 441], [188, 525], [154, 525], [136, 560], [158, 593], [191, 596], [233, 582]]




def get_outer_angle_4(p_left, p_right, center, radius):
    x1, y1 = p_left
    x4, y4 = p_right
    xc, yc = center
    ax = x1 - xc
    ay = y1 - yc
    bx = x4 - xc
    by = y4 - yc
    q1 = ax * ax + ay * ay
    q2 = q1 + ax * bx + ay * by
    k2 = 4/3 * (math.sqrt(2 * q1 * q2) - q2) / (ax * by - ay * bx)

    x2 = xc + ax - k2 * ay
    y2 = yc + ay + k2 * ax
    x3 = xc + bx + k2 * by                                 
    y3 = yc + by - k2 * bx

    return 0, (x2, y2), (x3, y3)


def get_cm_from_interpolation(pts, interpolation_factor): # 0 is circ
    center_circle = (205, 520)
    center_polygon = (205, 500)
    #center = center_circle[0] * (1 - interpolation_factor) + center_polygon[0] * interpolation_factor, center_circle[1] * (1 - interpolation_factor) + center_polygon[1] * interpolation_factor
    center = center_polygon
    #radius = max([3 * functions.euclid_dist(p, center) / 4 for p in pts])
    radius = 70
    tangents = []
    for i in range(len(pts)):
        p1 = pts[(i-1) % len(pts)]
        p2 = pts[(i+1) % len(pts)]
        pr = (p2[0] - p1[0], p2[1] - p1[1])
        alpha = math.atan2(pr[1], pr[0]) % math.pi
        tangents.append(alpha)

    dists = []
    for i in range(len(pts)):
        dist = functions.euclid_dist(pts[i], pts[(i+1) % len(pts)])
        d = dist * 0.3
        dists.append(d)

    pts_interpolated = [functions.project_to_circle(p, center_circle, radius) for p in pts]
    tangents_interpolated = []
    for i in range(len(pts)):
        x, y = pts[i]
        alpha = math.atan2(y - center_circle[1], x - center_circle[0])
        alpha -= 0.5 * math.pi
        alpha = alpha % math.pi
        tangents_interpolated.append(alpha)


    dists_interpolated = []
    for i in range(len(pts_interpolated)):
        x1, y1 = pts_interpolated[i]
        x4, y4 = pts_interpolated[(i+1) % len(pts_interpolated)]
        xc, yc = center_circle
        axx = x1 - xc
        ay = y1 - yc
        bx = x4 - xc
        by = y4 - yc
        q1 = axx * axx + ay * ay
        q2 = q1 + axx * bx + ay * by
        k2 = 4/3 * (math.sqrt(2 * q1 * q2) - q2) / (axx * by - ay * bx)
        d = math.sqrt((k2 * ay)**2 + (k2 * axx)**2)
        dists_interpolated.append(d)



    cm = multi.CurveContainer()
    cm.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=False, legend=False)
    pts = [[interpolation_factor * pp[0] + (1 - interpolation_factor) * pc[0], interpolation_factor * pp[1] + (1 - interpolation_factor) * pc[1]] for (pp, pc) in zip(pts, pts_interpolated)]

    new_tangents = []
    for tp, tc in zip(tangents, tangents_interpolated):
        if abs(tp - tc) > math.pi / 2:
            if tp < math.pi / 2:
                t = interpolation_factor * (tp + math.pi) + (1 - interpolation_factor) * tc
            elif tc < math.pi / 2:
                t = interpolation_factor * tp + (1 - interpolation_factor) * (tc + math.pi)
        else:
            t = interpolation_factor * tp + (1 - interpolation_factor) * tc

        new_tangents.append(t)

    tangents = new_tangents

    dists = [interpolation_factor * dp + (1 - interpolation_factor) * dc for (dp, dc) in zip(dists, dists_interpolated)]

    ctrlpts = []
    for i in range(len(pts)):
        c = BSpline.Curve()
        c.degree = 3
        d = dists[i]

        p1, p2 = get_points_on_tangent(pts[i], tangents[i], d)
        if math.atan2(p1[1] - center[1], p1[0] - center[0]) < math.atan2(p2[1] - center[1], p2[0] - center[0]):
            p1, p2 = p2, p1
        if functions.euclid_dist(p1, pts[(i+1) % len(pts)]) < functions.euclid_dist(p2, pts[(i+1) % len(pts)]):
            p2, p1 = p1, p2

        p3, p4 = get_points_on_tangent(pts[(i+1) % len(pts)], tangents[(i+1) % len(pts)], d)
        #if math.atan2(p3[1] - center[1], p3[0] - center[0]) < math.atan2(p4[1] - center[1], p4[0] - center[0]):
        #    p3, p4 = p4, p3

        #coords = [p1, p2, p3, p4]

        
        #cr = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        #p1, p2, p3, p4 = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, cr))[::-1]))) % 360)

        #p1, p2, p3, p4 = sorted([p1, p2, p3, p4], key=lambda q: (math.atan2(q[1] - center[1], q[0] - center[0])) % (2 * math.pi), reverse=True)

        if functions.euclid_dist(p4, pts[i]) < functions.euclid_dist(p3, pts[i]):
            p4, p3 = p3, p4

        
        #print(pts[i], p1, p2, p3, p4)

        c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
        c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        cm.add(c)

    return cm

#cm = get_cm_from_interpolation(pts, center, radius, INTERPOLATION_FACTOR)
#radius = max([3 * functions.euclid_dist(p, center) / 4 for p in pts])

#for i, f in enumerate([0, 0.25, 0.5, 0.75, 1]):
for f in np.arange(0, 1, 0.25):
    colors = ['red', 'green', 'blue', 'purple', 'black']
    cm = get_cm_from_interpolation(pts, f)

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


