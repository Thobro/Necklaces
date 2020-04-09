import tripy
import numpy as np
from tqdm import tqdm
import math

def triangle_area(t):
    """Compute the area of a triangle."""
    A, B, C = t[0], t[1], t[2]
    return abs(0.5 * (((B[0] - A[0]) * (C[1] - A[1])) - ((C[0] - A[0]) * (B[1] - A[1]))))

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
    return (center[0] - p[0])**2 + (center[1] - p[1])**2 <= radius**2

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

def sample_shape(s, n=50):
    """Samples a shape that consists of multiple polygons."""
    triangulation = []
    for polygon in s:
        tri = triangulate_polygon(polygon)
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
                for point_set in point_sets:
                    if len([p for p in point_set if in_circle(center, radius, p)]) < int(len(point_set) / 2):
                        save = False
                        break
                if save:
                    best = center, radius

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
                        for point_set in point_sets:
                            if len([p for p in point_set if in_circle(center, radius, p)]) < int(len(point_set) / 2):
                                save = False
                                break
                        if save:
                            best = center, radius

    return best


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