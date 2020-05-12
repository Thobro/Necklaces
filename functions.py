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

def point_in_traingle(p, p0, p1, p2):
    A = 1/2 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
    sign = -1 if A < 0 else 1
    s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]) * sign
    t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]) * sign
    
    return s > 0 and t > 0 and (s + t) < 2 * A * sign

def point_from_triangle(t):
    """Sample a point from a triangle."""
    A, B, C = t[0], t[1], t[2]
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    p = ( A[0] + x * (B[0] - A[0]) + y * (C[0] - A[0]), A[1] + x * (B[1] - A[1]) + y * (C[1] - A[1]) )

    if point_in_traingle(p, A, B, C):
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

def sample_points_in_water(triangulation, n, x_min, x_max, y_min, y_max):
    sample = []
    while len(sample) != n:
        x = x_min + np.random.uniform(0, 1) * (x_max - x_min)
        y = y_min + np.random.uniform(0, 1) * (y_max - y_min)
        p = (x, y)
        save = True
        for t in triangulation:
            A, B, C = t[0], t[1], t[2]
            if point_in_traingle(p, A, B, C):
                save = False
                break
        if save:
            sample.append((x, y))
    return sample

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
            if (op == 1 or op == 0) and d < c: # >=
                valid = False
                break
            '''elif op == 0 and d != c: # =
                valid = False
                break
            elif op == -1 and d > c: # <=
                valid = False
                break'''

        if valid:
            valid_configs.append(list(open_discs))


    circles = []
    for v in valid_configs: # O(n^2)
        x, y, r = make_circle(v) # O(n)
        valid = True
        for point_set, (c, op) in zip(point_sets, constraints): # O(n)
            if op == 1:
                continue
            no_points = len([p for p in point_set if euclid_dist(p, (x, y)) <= r])
            if op == 0 and no_points != c:
                valid = False
                break
            elif op == -1 and no_points > c:
                valid = False
                break
        if valid:
            circles.append((x, y, r))

    if len(circles) == 0:
        min_disc = (0, 0, math.inf)
    else:
        min_disc = min(circles, key=lambda c: c[2])
        
    return min_disc


def smallest_k_disc_fast(point_sets, constraints): # O(n^3)
    print("Computing smallest disc using the fast algorithm...")
    #constraints = [(int(len(point_set) / 2), 1) for point_set in point_sets]
    
    c_depth = [0 for ps in point_sets]
    all_points = [p for point_set in point_sets for p in point_set]
    
    mapping = {}
    for i in range(len(point_sets)):
        for p in point_sets[i]:
            mapping[p] = i

    r = 100000000000
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


def smallest_k_disc_facade(point_sets, water_constraint=25, region_constraint=25):
    constraints = [(int(len(point_set) / 2), 1) for point_set in point_sets]
    constraints[0] = (1, 1)

    if water_constraint < 100:
        constraints[-1] = (0, 1)
    else:
        point_sets.pop(len(point_sets) - 1)
    p, r = smallest_k_disc_fast(point_sets, constraints)

    water_count = len([q for q in point_sets[-1] if euclid_dist(q, p) <= r])
    print(water_count)


    # Multiple discs
    discs = [(p, r)]
    while True:
        print(discs)
        discard_candidates = []
        for i in range(len(point_sets) - 1):
            if len([q for q in point_sets[i] if int(euclid_dist(q, p)) == int(r)]) > 0:
                discard_candidates.append(i)

        min_dif = 0
        to_discard = None
        for index in discard_candidates: # Find the country that reduces the number of water points most quickly
            candidtate_sets = point_sets[:index] + point_sets[index+1:]
            new_constraints = [(int(len(point_set) / 2), 1) for point_set in candidtate_sets]
            #new_constraints[-1] = (0, 1)
            p1, r1 = smallest_k_disc_fast(candidtate_sets, new_constraints)
            #water_count = len([p for p in point_sets[-1] if euclid_dist(p, p1) <= r1])
            #print(water_count)
            if r - r1 > min_dif:
                p = p1
                r = r1
                to_discard = index

        point_sets.pop(to_discard)
        print(len(point_sets))
        if len(point_sets) <= region_constraint:
            discs.append((p1, r1))
            break

    return discs

    if water_count <= water_constraint:
        return p, r

    # While we still have too much water
    while True:
        discard_candidates = []
        for i in range(len(point_sets) - 1):
            if len([q for q in point_sets[i] if int(euclid_dist(q, p)) == int(r)]) > 0:
                discard_candidates.append(i)
        
        min_water = math.inf
        to_discard = None
        for index in discard_candidates: # Find the country that reduces the number of water points most quickly
            candidtate_sets = point_sets[:index] + point_sets[index+1:]
            new_constraints = [(int(len(point_set) / 2), 1) for point_set in candidtate_sets]
            new_constraints[-1] = (0, 1)
            p1, r1 = smallest_k_disc_fast(candidtate_sets, new_constraints)
            water_count = len([p for p in point_sets[-1] if euclid_dist(p, p1) <= r1])
            print(water_count)
            if water_count < min_water:
                p = p1
                r = r1
                to_discard = index
                min_water = water_count
                
        point_sets.pop(to_discard)
        discarded.append(to_discard)
        if min_water <= water_constraint:
            break
            
    print(p, r)
    return p, r


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