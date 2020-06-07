import numpy as np
from scipy import interpolate
import functions
import math
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from geomdl import multi

def get_points_on_tangent(p, a, dist):
    dx = math.cos(a) * dist
    dy = math.sin(a) * dist
    return (p[0] + dx, p[1] + dy), (p[0] - dx, p[1] - dy)

center = (317, 530)
cm = multi.CurveContainer()
cm.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=False, legend=False)
pts = [[485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]

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
    p1, p2 = sorted([p1, p2], key=lambda q: (math.atan2(q[1] - center[1], q[0] - center[0]) + 2 * math.pi) % (2 * math.pi), reverse=True)
    p3, p4 = get_points_on_tangent(pts[(i+1) % len(pts)], tangents[(i+1) % len(pts)], d)
    p3, p4 = sorted([p3, p4], key=lambda q: (math.atan2(q[1] - center[1], q[0] - center[0]) + 2 * math.pi) % (2 * math.pi), reverse=True)

    c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
    c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    cm.add(c)

cm.render()