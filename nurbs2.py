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
    
pts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604]]
cm = multi.CurveContainer()
cm.vis = VisMPL.VisCurve2D()
curve = NURBS.Curve()
curve.degree = 3 # order = degree + 1
curve.ctrlpts = pts
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
cm.add(curve)
c = fitting.interpolate_curve(pts, 3)
cm.add(c)
cm.render()
