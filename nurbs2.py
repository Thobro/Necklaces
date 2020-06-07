from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl.visualization import VisMPL
from geomdl import NURBS
from geomdl import multi
import functions
import math
from smallestenclosingcircle import make_circle
from geomdl import fitting
import numpy as np
from scipy import interpolate

def project_to_circle(point, center, radius):
    xp, yp = point
    xc, yc = center
    alpha = math.atan2(yp - yc, xp - xc)
    dx = radius * math.cos(alpha)
    dy = radius * math.sin(alpha)
    return (xc + dx, yc + dy)


def get_outer_angle(p1, p2, center, radius):
    xp1, yp1 = p1
    xp2, yp2 = p2
    xc, yc = center
    a = math.atan2(yp1 - yc, xp1 - xc)
    b = math.atan2(yp2 - yc, xp2 - xc)
    beta = abs((a - b) % (2 * math.pi))

    mid = (xp1 + xp2) / 2, (yp1 + yp2) / 2
    proj_1 = functions.euclid_dist(center, mid)
    alpha_a = functions.euclid_dist(mid, p2)
    proj_2 = alpha_a * math.tan(beta / 2)
    p3 = project_to_circle(mid, center, proj_1 + proj_2)

    return beta / 2, p3


def get_outer_angle_2(p1, p2, center, radius):
    xp1, yp1 = p1
    xp2, yp2 = p2
    xc, yc = center
    a = math.atan2(yp1 - yc, xp1 - xc)
    b = math.atan2(yp2 - yc, xp2 - xc)
    beta = abs((a - b) % (2 * math.pi))

    mid = (xp1 + xp2) / 2, (yp1 + yp2) / 2
    proj_1 = functions.euclid_dist(center, mid)
    alpha_a = functions.euclid_dist(mid, p2)
    proj_2 = alpha_a * math.tan(beta / 2)

    mid1 = (xp1 + mid[0]) / 2, (yp1 + mid[1]) / 2
    proj = radius / math.cos(beta / 4)
    p31 = project_to_circle(mid1, center, proj_1 + proj_2)

    mid2 = (xp2 + mid[0]) / 2, (yp2 + mid[1]) / 2
    proj = radius / math.cos(beta / 4)
    p32 = project_to_circle(mid2, center, proj_1 + proj_2)

    return beta / 2, p31, p32

def get_outer_angle_3(p_left, p_right, center, radius):
    xp1, yp1 = p_left
    xp2, yp2 = p_right
    xc, yc = center
    a = math.atan2(yp1 - yc, xp1 - xc)
    b = math.atan2(yp2 - yc, xp2 - xc)
    gamma = abs((a - b) % (2 * math.pi)) # Angle between left/right

    d = 0.5522 * radius
    print(d)

    beta = math.atan(d / radius)
    beta_left = a - beta
    beta_right = b + beta
    pl = center[0] + math.cos(beta_left), center[1] + math.sin(beta_left)
    pr = center[0] + math.cos(beta_right), center[1] + math.sin(beta_right)

    dist = math.sqrt(radius ** 2 + d ** 2)

    p1 = project_to_circle(pl, center, dist)
    p2 = project_to_circle(pr, center, dist)

    return beta / 2, p1, p2

def get_outer_angle_4(p_left, p_right, center, radius):
    #https://en.wikipedia.org/wiki/Composite_B%C3%A9zier_curve#Approximating_circular_arcs
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
    
#pts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
pts = [[485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
cm = multi.CurveContainer()
cm.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=True, legend=False)

x = [p[0] for p in pts]
y = [p[1] for p in pts]

# append the starting x,y coordinates
x = np.r_[x, x[0]]
y = np.r_[y, y[0]]

# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
tck, u = interpolate.splprep([x, y], s=0, per=True)

knots = list(tck[0])
coeff = list(tck[1])
deg = tck[2]
assert deg == 3
pts_int = [[x, y] for x, y in zip(coeff[0], coeff[1])]
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

c = NURBS.Curve()
c.degree = 3
c.ctrlpts = pts_int
c.knotvector = knots
'''c.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=False, legend=True)
c.render()'''

curves_0 = operations.decompose_curve(c)
base_pts = [(p[0], p[1]) for c in curves_0 for p in c.ctrlpts]
x, y, r = make_circle(base_pts) # O(n)
center = (x, y)
radius = r

pts_pr = []
for c in curves_0:
    pts_pr.append(project_to_circle(c.ctrlpts[0], center, radius))

curves_1 = []
for i in range(len(pts_pr) - 1):
    c = NURBS.Curve()
    c.degree = 3
    c.ctrlpts = [pts_pr[i], get_outer_angle_4(pts_pr[i], pts_pr[i + 1], center, radius)[1], get_outer_angle_4(pts_pr[i], pts_pr[i + 1], center, radius)[2], pts_pr[i + 1]]
    c.weights = [1, 1, 1, 1]
    c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    curves_1.append(c)

'''curves_3 = []
curves_4 = []
curves_5 = []


for c1, c2 in zip(curves_1, curves_0):
    c3 = NURBS.Curve()
    c4 = NURBS.Curve()
    c5 = NURBS.Curve()
    c3.degree = 3
    c4.degree = 3
    c5.degree = 3
    
    for i in range(4):
        p3 = (c1.ctrlpts[i][0] + c2.ctrlpts[i][0]) / 2, (c1.ctrlpts[i][1] + c2.ctrlpts[i][1]) / 2
        p5 = (c1.ctrlpts[i][0] + p3[0]) / 2, (c1.ctrlpts[i][1] + p3[1]) / 2
        p4 = (p3[0] + c2.ctrlpts[i][0]) / 2, (p3[1] + c2.ctrlpts[i][1]) / 2
        c3.ctrlpts.append(p3)
        c4.ctrlpts.append(p4)
        c5.ctrlpts.append(p5)

    c3.weights = [1, 1, 1, 1]
    c4.weights = [1, 1, 1, 1]
    c5.weights = [1, 1, 1, 1]
    c3.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    c4.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    c5.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(8):
        v3 = (c1.knotvector[i] + c2.knotvector[i]) / 2
        v4 = (v3 + c2.knotvector[i]) / 2
        v5 = (v3 + c1.knotvector[i]) / 2
        v3 = c1.knotvector[i]
        v4 = c1.knotvector[i]
        v5 = c1.knotvector[i]
        c3.knotvector.append(v3)
        c4.knotvector.append(v4)
        c5.knotvector.append(v5)

    
    curves_3.append(c3)
    curves_4.append(c4)
    curves_5.append(c5)'''



#for c in curves_1:
#    cm.add(c)
for c in curves_0: # Shape
    cm.add(c)
#for c in curves_3:
#    cm.add(c)
cm.render()
