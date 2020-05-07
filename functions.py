import tripy
import numpy as np
from tqdm import tqdm
import math
import random
import time
from smallestenclosingcircle import make_circle

def triangle_area(t):
    """Compute the area of a triangle."""
    A, B, C = t[0], t[1], t[2]
    return abs(0.5 * (((B[0] - A[0]) * (C[1] - A[1])) - ((C[0] - A[0]) * (B[1] - A[1]))))

def polygon_area(ts):
    return sum([triangle_area(t) for t in ts])

def point_from_triangle(t):
    """Sample a point from a triangle."""
    A, B, C = t[0], t[1], t[2]
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    p = ( A[0] + x * (B[0] - A[0]) + y * (C[0] - A[0]), A[1] + x * (B[1] - A[1]) + y * (C[1] - A[1]) )

    area = triangle_area(t)
    a1 = triangle_area([A, B, p])
    a2 = triangle_area([A, C, p])
    a3 = triangle_area([B, C, p])
    if round(a1 + a2 + a3, 6) == round(area, 6):
        return p
    else:    
        return ( A[0] + A[0] + (B[0] - A[0]) + (C[0] - A[0]) - p[0], A[1] + A[1] + (B[1] - A[1]) + (C[1] - A[1]) - p[1] )

def sq_euclid_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euclid_dist(p1, p2):
    return math.sqrt(sq_euclid_dist(p1, p2))

def points_to_circle(p1, p2, p3):
    """Compute centre and radius of circle of two points."""
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def in_circle(center, radius, p):
    """True if point lies in circle defined by center and radius."""
    return math.sqrt((center[0] - p[0])**2 + (center[1] - p[1])**2) <= radius

def triangulate_polygon(p):
    """Triangulates a polygon."""
    triangles = tripy.earclip(p)
    return triangles

def sample_polygon(p, n=50):
    """Samples a polygon."""
    triangulation = triangulate_polygon(p)
    polygon_size = sum([triangle_area(t) for t in triangulation])
    triangulation = [(t, triangle_area(t) / polygon_size) for t in triangulation]

    triangulation[0] = triangulation[0][0], triangulation[0][1], 0
    for i in range(1, len(triangulation)):
        triangulation[i] = (triangulation[i][0], triangulation[i][1] + triangulation[i-1][1], triangulation[i-1][1])

    samples = np.random.uniform(0, 1, n)
    points = []
    for s in samples:
        for t, ma, mi in triangulation:
            if mi <= s <= ma:
                p = point_from_triangle(t)
                points.append(p)

    return points

def sample_shape(s, r, n=50, tri=None, threshold=0):
    """Samples a shape that consists of multiple polygons."""
    triangulation = []
    if tri:
        for polygon in tri:
            if polygon_area(polygon) > threshold:
                for triangle in polygon:
                    triangulation.append(triangle)
    else:
        for polygon in s:
            tri = triangulate_polygon(polygon)
            if polygon_area(tri) > threshold:
                for triangle in tri:
                    triangulation.append(triangle)
    
    polygon_size = sum([triangle_area(t) for t in triangulation])
    triangulation = [(t, triangle_area(t) / polygon_size) for t in triangulation]

    triangulation[0] = triangulation[0][0], triangulation[0][1], 0
    for i in range(1, len(triangulation)):
        triangulation[i] = (triangulation[i][0], triangulation[i][1] + triangulation[i-1][1], triangulation[i-1][1])

    samples = np.random.uniform(0, 1, n)
    points = []
    for s in samples:
        for t, ma, mi in triangulation:
            if mi <= s <= ma:
                p = point_from_triangle(t)
                points.append(p)

    return points

def smallest_k_disc(point_sets):
    all_points = [p for point_set in point_sets for p in point_set]
    constraints = [int(len(point_set) / 2) for point_set in point_sets]

    best = (None, None), math.inf
    print("Computing all discs with 2 points...")
    for i in tqdm(range(len(all_points))):
        for j in range(len(all_points)):
            if i == j:
                continue
            radius = math.sqrt(sq_euclid_dist(all_points[i], all_points[j])) / 2
            
            center = ((all_points[i][0] + all_points[j][0]) / 2, (all_points[i][1] + all_points[j][1]) / 2)
            
            if radius < best[1]:
                save = True
                for l in range(len(point_sets)):
                    if len([p for p in point_sets[l] if in_circle(center, radius, p)]) < constraints[l]:
                        save = False
                        break
                if save:
                    best = center, radius

    print("Smallest disc: ")
    print(best)
    print()

    print("Computing all discs with 3 points...")
    for i in tqdm(range(len(all_points))):
        for j in range(len(all_points)):
            if sq_euclid_dist(all_points[i], all_points[j]) >= best[1]**2:
                continue
            for k in range(len(all_points)):
                if i != j and j != k:
                    center, radius = points_to_circle(all_points[i], all_points[j], all_points[k])
                    if radius < best[1]:
                        save = True
                        for l in range(len(point_sets)):
                            if len([p for p in point_sets[l] if in_circle(center, radius, p)]) < constraints[l]:
                                save = False
                                break
                        if save:
                            best = center, radius

    print("Smallest disc: ")
    print(best)
    print()
    return best

def test_rp_gt_r(r, p, constraints, all_points, mapping):
    """True if rp >= r."""
    #print(f"Computing rp for {p}...")
    all_points = [p for point_set in point_sets for p in point_set]
    all_points = [q for q in all_points if q != p and euclid_dist(p, q) < 2 * r] # Compute I(p, r)
    c_depth = [0 for ps in point_sets]
    open_discs = [p]
    valid_configs = []

    intersections = []

    c_depth[mapping[p]] += 1

    intersection_mapping = {}

    for q in all_points:
        midpoint = ((p[0] + q[0]) / 2), ((p[1] + q[1]) / 2)
        a = euclid_dist(p, q) / 2
        h = math.sqrt(r**2 - a**2)

        p1 = (midpoint[0] + h * ((q[1] - p[1]) / (2 * a)), midpoint[1] - h * ((q[0] - p[0]) / (2 * a)))
        p2 = (midpoint[0] - h * ((q[1] - p[1]) / (2 * a)), midpoint[1] + h * ((q[0] - p[0]) / (2 * a)))

        intersection_mapping[p1] = q
        intersection_mapping[p2] = q

        intersections.append(p1)
        intersections.append(p2)

        if abs((math.atan2(p1[1] - p[1], p1[0] - p[0]) + 2 * math.pi) % (2 * math.pi) - (math.atan2(p2[1] - p[1], p2[0] - p[0]) + 2 * math.pi) % (2 * math.pi)) > math.pi:
            open_discs.append(q)
            c_depth[mapping[q]] += 1

    intersections = sorted(intersections, key=lambda q: (math.atan2(q[1] - p[1], q[0] - p[0]) + 2 * math.pi) % (2 * math.pi))

    for i in range(0, len(intersections)):
        intersection = intersections[i]
        if intersection_mapping[intersection] in open_discs:
            c_depth[mapping[intersection_mapping[intersection]]] -= 1
            open_discs.remove(intersection_mapping[intersection])
        else:
            c_depth[mapping[intersection_mapping[intersection]]] += 1
            open_discs.append(intersection_mapping[intersection])

        if all([c_depth[i] >= constraints[i] for i in range(len(c_depth))]):
            valid_configs.append(list(open_discs))

    return len(valid_configs) == 0

def find_rp(r, p, constraints, point_sets, mapping):
    """Function returns the smallest disc that contains p on its boundary."""
    #print(f"Computing rp for {p}...")
    all_points = [p for point_set in point_sets for p in point_set]
    all_points = [q for q in all_points if q != p and euclid_dist(p, q) < 2 * r] # Compute I(p, r)
    c_depth = [0 for ps in point_sets]
    open_discs = [p]
    valid_configs = []

    intersections = []

    c_depth[mapping[p]] += 1

    intersection_mapping = {}

    for q in all_points:
        midpoint = ((p[0] + q[0]) / 2), ((p[1] + q[1]) / 2)
        a = euclid_dist(p, q) / 2
        h = math.sqrt(r**2 - a**2)

        p1 = (midpoint[0] + h * ((q[1] - p[1]) / (2 * a)), midpoint[1] - h * ((q[0] - p[0]) / (2 * a)))
        p2 = (midpoint[0] - h * ((q[1] - p[1]) / (2 * a)), midpoint[1] + h * ((q[0] - p[0]) / (2 * a)))

        intersection_mapping[p1] = q
        intersection_mapping[p2] = q

        intersections.append(p1)
        intersections.append(p2)

        if abs((math.atan2(p1[1] - p[1], p1[0] - p[0]) + 2 * math.pi) % (2 * math.pi) - (math.atan2(p2[1] - p[1], p2[0] - p[0]) + 2 * math.pi) % (2 * math.pi)) > math.pi:
            open_discs.append(q)
            c_depth[mapping[q]] += 1

    intersections = sorted(intersections, key=lambda q: (math.atan2(q[1] - p[1], q[0] - p[0]) + 2 * math.pi) % (2 * math.pi))

    for i in range(0, len(intersections)):
        intersection = intersections[i]
        if intersection_mapping[intersection] in open_discs:
            c_depth[mapping[intersection_mapping[intersection]]] -= 1
            open_discs.remove(intersection_mapping[intersection])
        else:
            c_depth[mapping[intersection_mapping[intersection]]] += 1
            open_discs.append(intersection_mapping[intersection])

        valid = True
        for d, (c, op) in zip(c_depth, constraints):
            if op == 1 and d < c: # >=
                valid = False
                break
            elif op == 0 and d != c: # =
                valid = False
                break
            elif op == -1 and d > c: # <=
                valid = False
                break

        if valid:
            valid_configs.append(list(open_discs))

    circles = [make_circle(v) for v in valid_configs] # O(n^2)
    if len(circles) == 0:
        min_disc = (0, 0, math.inf)
    else:
        min_disc = min(circles, key=lambda c: c[2])
        
    return min_disc


def smallest_k_disc_fast(point_sets): # O(n^3)
    print("Computing smallest disc using the fast algorithm...")
    constraints = [(int(len(point_set) / 2), 1) for point_set in point_sets]
    constraints[0] = (11, 0)
    c_depth = [0 for ps in point_sets]
    all_points = [p for point_set in point_sets for p in point_set]
    
    mapping = {}
    for i in range(len(point_sets)):
        for p in point_sets[i]:
            mapping[p] = i

    r = 100000000
    s = 0

    min_disc = (0, 0, math.inf)
    for p in tqdm(all_points):
        s += 1
        while (min_cand := find_rp(r, p, constraints, point_sets, mapping))[2] < r:
            s += 1
            r = min_cand[2]
            min_disc = min_cand

    print(s)
    print("Smallest disc: ")
    print(min_disc)
    print()

    return (min_disc[0], min_disc[1]), min_disc[2]


def smallest_k_disc_fast_randomised(point_sets): # O(n^3)
    print("Computing smallest disc using the fast algorithm...")
    all_points = [p for point_set in point_sets for p in point_set]
    constraints = [int(len(point_set) / 2) for point_set in point_sets]
    c_depth = [0 for ps in point_sets]
    mapping = {}
    for i in range(len(point_sets)):
        for p in point_sets[i]:
            mapping[p] = i

    r = 100000000000
    s = 0

    min_disc = (0, 0, math.inf)
    p = random.choice(all_points)
    while (min_cand := find_rp(r, p, constraints, point_sets, mapping))[2] < r:
        r = min_cand[2]
        min_disc = min_cand

    candidates = list(all_points)

    while len((candidates := [q for q in candidates if not test_rp_gt_r(r, q, constraints, point_sets, mapping)])) != 0:
        print(len(candidates))
        p = random.choice(candidates)
        min_disc = (0, 0, math.inf)
        while (min_cand := find_rp(r, p, constraints, candidates, mapping))[2] < r:
            r = min_cand[2]
            min_disc = min_cand


    print(s)
    print("Smallest disc: ")
    print(min_disc)
    print()
    return (min_disc[0], min_disc[1]), min_disc[2]
    

def shape_to_parts(polygon):
    polygons = []
    if len(polygon.parts) == 1:
        return [polygon.points]
    for i in range(1, len(polygon.parts)):
        start = polygon.parts[i - 1]
        end = polygon.parts[i]
        polygons.append(polygon.points[start:end])
    polygons.append(polygon.points[polygon.parts[-1]:])
    return polygons

def borders(shape1, shape2):
    all_points1 = {p for polygon in shape1 for p in polygon}
    all_points2 = {p for polygon in shape2 for p in polygon}

    intersection = all_points1.intersection(all_points2)
    return len(intersection) > 0