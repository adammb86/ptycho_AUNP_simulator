import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def generate_random_points(num_points):
    """
    Generate a set of random points in 2D space.
    """
    return np.random.rand(num_points, 2)

def compute_polygon_area(points):
    """
    Compute the area of a polygon given its vertices.
    """
    hull = ConvexHull(points)
    return hull.volume

def generate_polygon(min_area, max_attempts=1000):
    """
    Generate a polygon with area at least as large as min_area.
    """
    for _ in range(max_attempts):
        points = generate_random_points(num_points=8)
        if compute_polygon_area(points) >= min_area:
            return points
    raise ValueError('Failed to generate a polygon with the desired area after max_attempts')

def plot_polygon(points):
    """
    Plot the polygon given its vertices.
    """
    hull = ConvexHull(points)
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.show()

# Test the function
polygon = generate_polygon(min_area=0.8)
plot_polygon(polygon)