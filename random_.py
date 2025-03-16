import random 
import os
import pickle as pkl
import numpy as np
import math
import pandas as pd

main_file = os.path.join(os.path.dirname(__file__))

# def generate_random_points(n):
#     points = [random.uniform(0, 1) for _ in range(n)]
#     return points
def generate_random_points(n, start=0, end=1036.8, min_dist=50, max_dist=60):
    points = []
    current_point = random.uniform(start, start + max_dist)
    points.append(current_point)
    
    while len(points) < n:
        step = random.uniform(min_dist, max_dist)
        next_point = points[-1] + step
        if next_point > end:
            break
        points.append(next_point)
    print(points)

    return points

def map_to_unit_interval(points, max_value):
    return [point / max_value for point in points]

# 假設N是10
with open('point.xlsx', 'rb') as f:
    data = pd.read_excel(f)

length = data['總距離'][0]
print(length)

N = [6, 12, 18]
for point in N:
    random_points = generate_random_points(n = point, end = length)
    random_points = map_to_unit_interval(random_points, length)

    rand_path = os.path.join(main_file, 'random_storage', str(point)+'.pkl')
    print('--------------------------------')
    print("mapping back:", random_points)
    print(np.asarray(random_points).shape)
    print('--------------------------------')
    with open(rand_path, 'wb') as f:
        pkl.dump(random_points, f)
