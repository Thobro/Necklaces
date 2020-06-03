from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl.visualization import VisMPL
from geomdl import NURBS
from geomdl import multi
import functions
import math

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
    alpha_a = functions.euclid_dist(mid, p1)
    proj_2 = alpha_a * math.tan(beta / 2)
    p3 = project_to_circle(mid, center, proj_1 + proj_2)

    return beta / 2, p3
    
curve = NURBS.Curve()
curve2 = NURBS.Curve()
curve3 = NURBS.Curve()

center = (317, 465)

pts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
radius = max([functions.euclid_dist(center, p) for p in pts])
pts_pr = [project_to_circle(p, center, radius) for p in pts]

pts_se = [(p, project_to_circle(p, center, radius)) for p in pts]

for i in range(len(pts_pr) - 1):
    print(pts_pr[i], pts_pr[i+1], get_outer_angle(pts_pr[i], pts_pr[i+1], center, radius))


#project_to_circle(pts[0], center, radius)
#pts = [project_to_circle(p, center, radius) for p in pts]
# Set up the curve
curve.degree = 2 # order = degree + 1
curve.ctrlpts = pts
curve2.degree = 2
curve2.ctrlpts = pts_pr
curve3.degree = 2

mcrv = multi.CurveContainer()
mcrv.delta = 0.01

pts_circle = []
weights = []
for i in range(len(pts_pr) - 1):
    pts_circle.append([pts_pr[i], get_outer_angle(pts_pr[i], pts_pr[i + 1], center, radius)[1], pts_pr[i + 1]])
    weights.append([1, math.cos(get_outer_angle(pts_pr[i], pts_pr[i + 1], center, radius)[0]), 1])

    c = NURBS.Curve()
    c.degree = 2
    c.delta = 0.01
    c.ctrlpts = [pts_pr[i], get_outer_angle(pts_pr[i], pts_pr[i + 1], center, radius)[1], pts_pr[i + 1]]
    c.weights = [1, math.cos(get_outer_angle(pts_pr[i], pts_pr[i + 1], center, radius)[0]), 1]
    c.knotvector = [0,0,0,1,1,1]
    mcrv.add(c)

mcrv.vis = VisMPL.VisCurve2D(axes=False, labels=False, legend=False)
mcrv.render()
pass

curve3.ctrlpts = pts_circle
curve3.weights = weights
# Auto-generate knot vector
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts), clamped=False)
curve2.knotvector = utilities.generate_knot_vector(curve2.degree, len(curve2.ctrlpts), clamped=False)
curve3.knotvector = utilities.generate_knot_vector(curve3.degree, len(curve3.ctrlpts))

# Set evaluation delta
curve.delta = 0.01
curve2.delta = 0.01
curve3.delta = 0.01
curve.evaluate()
curve2.evaluate()
curve3.evaluate()
#operations.refine_knotvector(curve, [1])
#curve_list = operations.decompose_curve(curve)
#curves = multi.CurveContainer(curve_list)
#curves.vis = VisMPL.VisCurve2D()
#curves.render()

# Plot the control point polygon and the evaluated curve
#curve.vis = VisMPL.VisCurve2D(ctrlpts=False, axes=False, labels=False, legend=False)
curve.vis = VisMPL.VisCurve2D()
curve.vis.add(curve.evalpts, "evalpts", color='blue', idx=0)
#curve.vis.add(curve.ctrlpts, "ctrlpts", color='blue', idx=0)
curve.vis.add(curve2.evalpts, "evalpts", color='blue', idx=1)
#curve.vis.add(curve2.ctrlpts, "ctrlpts", color='blue', idx=1)
curve.vis.add(curve3.evalpts, "evalpts", color='green', idx=2)
#curve.vis.add(curve3.ctrlpts, "ctrlpts", color='green', idx=2)
curve.vis.render()
#curve.animate()
#curve.render()
#curve.render(filename="africa_curve.pdf", plot=False)