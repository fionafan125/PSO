import numpy as np
from scipy.interpolate import interp1d

def get_straight_dist(dist_points_meters, actual_length):
    init = np.array([0, 0])
    distances = np.linalg.norm(dist_points_meters - init, axis=1)

    acc_dist = distances.copy()
    for i in range(1, len(distances)):
        acc_dist[i] = acc_dist[i-1] + distances[i]
    dilation = acc_dist[-1] / actual_length
    distances = distances / dilation
    acc_dist = acc_dist / dilation
    return distances, acc_dist

def map_back_to_curve(straight_distances, original_points, actual_length):
    distances, acc_dist = get_straight_dist(original_points, actual_length)
    interp_x = interp1d(acc_dist, original_points[:, 0], kind='linear')
    interp_y = interp1d(acc_dist, original_points[:, 1], kind='linear')
    curve_x = interp_x(straight_distances)
    curve_y = interp_y(straight_distances)
    curve_points = np.vstack((curve_x, curve_y)).T
    return curve_points

# 示例使用
dist_points_meters = np.array([
    [0, 0],
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
])
actual_length = 10
straight_distances, _ = get_straight_dist(dist_points_meters, actual_length)

# 生成一些直线上的点
straight_line_points = np.linspace(0, actual_length, len(dist_points_meters))

print(straight_line_points)
# 将直线上的点映射回原曲线
mapped_points = map_back_to_curve(straight_line_points, dist_points_meters, actual_length)

print("Original points:")
print(dist_points_meters)
print("Mapped back points:")
print(mapped_points)
