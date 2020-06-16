import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import functions
import math

CIRCLE = False


#pts = [[485, 475], [387, 577]]

pts = [[485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [204, 604], [284, 612], [387, 577]]
#pts = [[100, 0], [0, -100], [-100, 0], [0, 100]]
center = (317, 465)
radius = max([functions.euclid_dist(p, center) for p in pts])

if CIRCLE:
    pts = [functions.project_to_circle(p, center, radius) for p in pts]
    circle_pts = [ [center[0] + radius, center[1]], [center[0] - radius, center[1]], [center[0], center[1] + radius], [center[0], center[1] - radius] ]
    circle_pts_2 = [ [center[0] + radius, center[1] + radius], [center[0] - radius, center[1] - radius], [center[0] - radius, center[1] + radius], [center[0] + radius, center[1] - radius] ]
    circle_pts_2 = [functions.project_to_circle(p, center, radius) for p in circle_pts_2]
    pts.extend(circle_pts)
    pts.extend(circle_pts_2)

pts = sorted(pts, key=lambda q: (math.atan2(q[1] - center[1], q[0] - center[0]) + 2 * math.pi) % (2 * math.pi))

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
pts = [[x, y] for x, y in zip(coeff[0], coeff[1])]

# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
# plot the result
fig, ax = plt.subplots(1, 1)
ax.axis('equal')
ax.plot(x, y, 'or')
ax.plot(coeff[0], coeff[1], 'ob')
ax.plot(xi, yi, '-b')
plt.show()