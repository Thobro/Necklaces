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
import shape_read

FILENAME_HIGHRES = "Admin_1_10/ne_10m_admin_1_states_provinces_lakes.shp"

fig = plt.figure(1, figsize=(7,7))
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

def plot_shape_recs(shape_recs, color=True):
    if color:
        color_mapping = get_colors_from_shape_recs(shape_recs)

    for shape, record in shape_recs:
        for polygon in shape:
            poly = Polygon(polygon)
            x,y = poly.exterior.xy
            ax.plot(x, y, color='000', alpha=1,
                linewidth=1, zorder=0)
            if color:
                ax.fill(x, y, color=map_colors[record['mapcolor13'] - 1], alpha=1,
                    linewidth=0, zorder=0)
            else:
                ax.fill(x, y, color=(1, 1, 1), alpha=1,
                    linewidth=0, zorder=0)

geo = []
#shape_recs = shape_read.shapefile_to_shape_recs(FILENAME_HIGHRES)
#shape_recs = [(shape, rec) for (shape, rec) in shape_recs if rec['admin'] == 'Netherlands' and rec['scalerank'] <= 8]
#plot_shape_recs(shape_recs, color=False)

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


def get_curve_length(c):
    l = 0
    for i in range(len(c.evalpts) - 1):
        l += functions.euclid_dist(c.evalpts[i], c.evalpts[i + 1])
    
    return l

#pts = [[274, 540], [252, 483], [209, 441], [188, 525], [154, 525], [136, 560], [158, 593], [191, 596], [233, 582]]
#center = (208, 525)
#radius = 70
pts = [[430, 518], [392, 430], [258, 213], [193, 528], [213, 595], [330, 580]]
radius = 115
center = (270, 476)
#pts = [[373, 457], [372, 400], [217, 394], [288, 433]]
#pts = [[717*10**3, 6.45*10**6], [591*10**3, 6.451*10**6], [302*10**3, 6.56*10**6], [424*10**3, 6.94*10**6], [473*10**3, 7.07*10**6], [836*10**3, 7.10*10**6], [890*10**3, 7.03*10**6]]
#center = (582994, 6834740)
#radius = 230565

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


def get_cm_from_interpolation_radial(pts, interpolation_factor): # 0 is circ
    center_circle = (200, 540)
    center_polygon = (223, 555)
    center = center_circle[0] * (1 - interpolation_factor) + center_polygon[0] * interpolation_factor, center_circle[1] * (1 - interpolation_factor) + center_polygon[1] * interpolation_factor
    radius = 70
    tangents = []
    for i in range(len(pts)):
        p1 = pts[(i-1) % len(pts)]
        p2 = pts[(i+1) % len(pts)]
        pr = (p2[0] - p1[0], p2[1] - p1[1])
        alpha = math.atan2(pr[1], pr[0])
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

        if functions.euclid_dist(p4, pts[i]) < functions.euclid_dist(p3, pts[i]):
            p4, p3 = p3, p4

        c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
        c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        cm.add(c)

    return cm


def get_cm_from_interpolation_length(pts, interpolation_factor, center_circle, radius): # 0 is circ
    center = center_circle
    cm = multi.CurveContainer()
    cm.vis = VisMPL.VisCurve2D(axes=False, labels=False, ctrlpts=False, legend=False)
    circle_length = 2 * math.pi * radius
    tangents = []
    curve_lengths = []
    for i in range(len(pts)):
        p1 = pts[(i-1) % len(pts)]
        p2 = pts[(i+1) % len(pts)]
        pr = (p2[0] - p1[0], p2[1] - p1[1])
        alpha = math.atan2(pr[1], pr[0]) % math.pi
        tangents.append(alpha)

    dists = []
    for i in range(len(pts)):
        dist = functions.euclid_dist(pts[i], pts[(i+1) % len(pts)])
        d = dist * 0.25
        dists.append(d)

    for i in range(len(pts)):
        c = BSpline.Curve()
        c.degree = 3
        d = dists[i]

        p1, p2 = get_points_on_tangent(pts[i], tangents[i], d)
        if functions.euclid_dist(p1, pts[(i+1) % len(pts)]) < functions.euclid_dist(p2, pts[(i+1) % len(pts)]):
            p2, p1 = p1, p2

        p3, p4 = get_points_on_tangent(pts[(i+1) % len(pts)], tangents[(i+1) % len(pts)], d)

        if functions.euclid_dist(p4, pts[i]) < functions.euclid_dist(p3, pts[i]):
            p4, p3 = p3, p4

        c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
        c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]

        length = get_curve_length(c)
        curve_lengths.append(length)
        cm.add(c)


    curve_lengths.reverse()
    xc, yc = center_circle
    pts_r = list(pts)
    pts_r.reverse()
    distance_factor = circle_length / sum(curve_lengths)
    pts_interpolated = []
    starting_point = functions.project_to_circle(pts[0], center_circle, radius)
    pts_interpolated.append(starting_point)
   
    starting_angle = math.atan2(starting_point[1] - yc, starting_point[0] - xc)
    alpha = curve_lengths[0] * distance_factor / radius
    
    xp = xc + radius * math.cos(alpha)
    yp = yc + radius * math.sin(alpha)

    angle_sum = starting_angle
    for i in range(len(pts) - 1):
        alpha = curve_lengths[i] * distance_factor / radius
        angle_sum += alpha
        xp = xc + radius * math.cos(angle_sum)
        yp = yc + radius * math.sin(angle_sum)
        pts_interpolated.append((xp, yp))

    pts_interpolated.reverse()
    pts_interpolated.insert(0, pts_interpolated.pop())
    
    tangents_interpolated = []
    for i in range(len(pts_interpolated)):
        x, y = pts_interpolated[i]
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

    angle_mapping = {}

    ctrlpts = []
    for i in range(len(pts)):
        c = BSpline.Curve()
        c.degree = 3
        d = dists[i]

        p1, p2 = get_points_on_tangent(pts[i], tangents[i], d)

        if i == 0:
            print(tangents[i])
            if p1[1] < p2[1] and tangents[i] < math.pi:
                p1, p2 = p2, p1
            elif p1[1] > p2[1] and tangents[i] > math.pi:
                p1, p2 = p2, p1
        
        else:

            if math.atan2(p1[1] - center[1], p1[0] - center[0]) < math.atan2(p2[1] - center[1], p2[0] - center[0]):
                p1, p2 = p2, p1
            if functions.euclid_dist(p1, pts[(i+1) % len(pts)]) < functions.euclid_dist(p2, pts[(i+1) % len(pts)]):
                p2, p1 = p1, p2

        xp, yp = pts[i]
        if (xp, yp) in angle_mapping:
            p = angle_mapping[(xp, yp)]

            if p[0] > xp and p2[0] > xp or p[0] < xp and p2[0] < xp:
                p1, p2 = p2, p1

        angle_mapping[(xp, yp)] = p2

        p3, p4 = get_points_on_tangent(pts[(i+1) % len(pts)], tangents[(i+1) % len(pts)], d)

        if functions.euclid_dist(p4, pts[i]) < functions.euclid_dist(p3, pts[i]):
            p4, p3 = p3, p4

        xp, yp = pts[(i+1) % len(pts)]
        angle_mapping[(xp, yp)] = p3

        c.ctrlpts = [pts[i], p2, p3, pts[(i+1) % len(pts)]]
        c.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        cm.add(c)

    return cm

for i, f in enumerate([0, 0.25, 0.5, 0.75, 1]):
    colors = ['red', 'green', 'blue', 'purple', 'black']
    cm = get_cm_from_interpolation_length(pts, f, center, radius)

    for j, c in enumerate(cm):
        curvepts = np.array(c.evalpts)
        ctrlpts = np.array(c.ctrlpts)
        get_curve_length(c)
        #cppolygon, = plt.plot(ctrlpts[:, 0], ctrlpts[:, 1], color='black', linestyle='-.', marker='o', markersize='3')
        curveplt, = plt.plot(curvepts[:, 0], curvepts[:, 1], color=colors[j % 5], linestyle='-')  # evaluated curve points

poly = Polygon(geo)
x, y = poly.exterior.xy
ax.plot(x, y, color='000', alpha=1,
        linewidth=1, zorder=0)
plt.show()

#cm.render()


