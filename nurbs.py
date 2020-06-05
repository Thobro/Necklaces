from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl.visualization import VisMPL
from geomdl import NURBS
from geomdl import multi
import functions
import math
from smallestenclosingcircle import make_circle

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
    
curve = NURBS.Curve()
mcrv_circle = multi.CurveContainer()
mcrv_circle.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=False, legend=False)
center = (317, 465)
#pts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
pts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604]]
pts_c = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
#pts = [[204, 604], [284, 612], [430, 366], [274, 438], [193, 436], [204, 604]]
#pts_c = [[204, 604], [284, 612], [430, 366], [274, 438], [193, 436], [204, 604], [284, 612], [430, 366]]
#radius = max([functions.euclid_dist(center, p) for p in pts])


curves_1 = []
curves_0 = []

curve.degree = 3 # order = degree + 1
curve.ctrlpts = pts_c
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts), clamped=False)
curve_list = operations.decompose_curve(curve)
knot_vectors = []
radius = 0



for c in curve_list:
    knot_vectors.append(c.knotvector)
    curves_0.append(c)
    radius = max(radius, max([functions.euclid_dist(center, p) for p in c.ctrlpts]))

base_pts = [(p[0], p[1]) for c in curve_list for p in c.ctrlpts]
x, y, r = make_circle(base_pts) # O(n)
center = (x, y)
radius = r


pts_pr = []
for c in curves_0:
    pts_pr.append(project_to_circle(c.ctrlpts[0], center, radius))
    #print(project_to_circle(c.ctrlpts[0], center, radius))

#print(len(pts_pr))
pts_pr[0] = pts_pr[-1]

circle_ctrl = []
for i in range(len(pts_pr) - 1):
    c = NURBS.Curve()
    c.degree = 3
    c.ctrlpts = [pts_pr[i], get_outer_angle_4(pts_pr[i], pts_pr[i + 1], center, radius)[1], get_outer_angle_4(pts_pr[i], pts_pr[i + 1], center, radius)[2], pts_pr[i + 1]]
    circle_ctrl.extend(c.ctrlpts)
    c.weights = [1, 1, 1, 1]
    c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    curves_1.append(c)


'''for i in range(0, len(all_pts) - 1, 4):
    c = NURBS.Curve()
    c.degree = 3
    ctrl = []
    for k in range(4):
        p = (all_pts[i+k][0] + circle_ctrl[i+k][0]) / 2, (all_pts[i+k][1] + circle_ctrl[i+k][1]) / 2
        c.ctrlpts.append(p)
    #c.ctrlpts = [((all_pts[i][0] + pts_pr[i][0]) / 2, ), all_pts[i+1], all_pts[i+2], all_pts[i+3]]
    c.weights = [1, 1, 1, 1]
    c.knotvector = utilities.generate_knot_vector(c.degree, len(c.ctrlpts))
    mcrv_circle.add(c)'''


for c in curves_1:
    mcrv_circle.add(c)
for c in curves_0: # Shape
    mcrv_circle.add(c)

curves_3 = []
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
    '''for i in range(8):
        v3 = (c1.knotvector[i] + c2.knotvector[i]) / 2
        v4 = (v3 + c2.knotvector[i]) / 2
        v5 = (v3 + c1.knotvector[i]) / 2
        v3 = c1.knotvector[i]
        v4 = c1.knotvector[i]
        v5 = c1.knotvector[i]
        c3.knotvector.append(v3)
        c4.knotvector.append(v4)
        c5.knotvector.append(v5)'''

    
    curves_3.append(c3)
    curves_4.append(c4)
    curves_5.append(c5)

curves_3[0].ctrlpts = [curves_3[-1].ctrlpts[-1], curves_3[0].ctrlpts[1], curves_3[0].ctrlpts[2], curves_3[0].ctrlpts[3]]
curves_4[0].ctrlpts = [curves_4[-1].ctrlpts[-1], curves_4[0].ctrlpts[1], curves_4[0].ctrlpts[2], curves_4[0].ctrlpts[3]]
curves_5[0].ctrlpts = [curves_5[-1].ctrlpts[-1], curves_5[0].ctrlpts[1], curves_5[0].ctrlpts[2], curves_5[0].ctrlpts[3]]

for c in curves_3:
    mcrv_circle.add(c)
for c in curves_4:
    mcrv_circle.add(c)
for c in curves_5:
    mcrv_circle.add(c)


mcrv_circle.render()
#mcrv_int.render()

#curve.evaluate()
#operations.refine_knotvector(curve, [1])
#curve_list = operations.decompose_curve(curve)
#curves = multi.CurveContainer(curve_list)
#curves.vis = VisMPL.VisCurve2D()
#curves.render()

# Plot the control point polygon and the evaluated curve
#curve.vis = VisMPL.VisCurve2D(ctrlpts=False, axes=False, labels=False, legend=False)
#curve.vis = VisMPL.VisCurve2D()
#curve.vis.add(curve.evalpts, "evalpts", color='blue', idx=0)
#curve.vis.add(curve.ctrlpts, "ctrlpts", color='blue', idx=0)
#curve.vis.render()
#curve.render(filename="africa_curve.pdf", plot=False)