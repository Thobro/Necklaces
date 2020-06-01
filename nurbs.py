from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl.visualization import VisMPL

# Create a B-Spline curve
curve = BSpline.Curve()

# Set up the curve
curve.degree = 2 # order = degree + 1
curve.ctrlpts = [[204, 604], [284, 612], [387, 577], [485, 475], [430, 366], [325, 231], [274, 438], [193, 436], [150, 525], [207, 603]]

# Auto-generate knot vector
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

# Set evaluation delta
#curve.delta = 0.01
operations.refine_knotvector(curve, [1])

# Plot the control point polygon and the evaluated curve
curve.vis = VisMPL.VisCurve2D()
curve.render()