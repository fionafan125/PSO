
import os
import matplotlib.pyplot as plt
import seaborn
import pickle as pkl
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
import numpy as np
import math
import random
import pandas as pd
from math import cos, radians
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import argparse
import os


def map_points_to_range(points, actual_length):
    mapped_points = [point * actual_length for point in points]
    return np.asarray(mapped_points)

def custom_scientific_format(num):
    if num == 0:
        return "0"
    exponent = int(f"{num:e}".split('e')[-1])  # 获取指数部分
    return f"1e{exponent}"

def plot_converge_curve(record_cost, storage_name):
    
    # 繪製折線圖
    iters = np.arange(1, 2001, 1)
    plt.plot(iters, record_cost, marker='o')  # 使用圓圈標記每個數據點

    # 標題和軸標籤
    plt.title("Simple Line Chart Example")
    plt.xlabel("Iters")
    plt.ylabel("Records")

    # 顯示圖表
    plt.savefig(storage_name)

def compute_dis_xy(init_pos, final_pos, meter_per_degree_latitude):
    

    # 計算南北方向上的距離（Δ緯度 × 每度緯度的公尺數）
    delta_latitude = final_pos[1] - init_pos[1]
    distance_north_south = delta_latitude * meter_per_degree_latitude

    # 計算東西方向上的距離（Δ經度 × cos(平均緯度) × 每度緯度的公尺數）
    average_latitude = radians((init_pos[1] + final_pos[1]) / 2)
    delta_longitude = final_pos[0] - init_pos[0]
    distance_east_west = delta_longitude * cos(average_latitude) * meter_per_degree_latitude
    return [distance_north_south, (-1)*distance_east_west]

def get_compared_dist(points, meters_per_degree_latitude):

    compared_dist = []
    for point in points:
        tmp_dist = compute_dis_xy(points[0,:], point, meters_per_degree_latitude)
        compared_dist.append(tmp_dist)

    compared_dist = np.array(compared_dist)
    return compared_dist

def get_straight_dist(dist_points_meters, actual_length):
    init = np.array([0, 0])
    distances = np.linalg.norm(dist_points_meters - init, axis=1)

    acc_dist = distances.copy()
    for i in range(1, len(distances)):
        acc_dist[i] = acc_dist[i-1] + acc_dist[i]
    dilation = acc_dist[len(acc_dist)-1] / actual_length
    distances = distances / dilation
    acc_dist = acc_dist / dilation
    return distances, acc_dist

def earned_money_count(input):
    global length, maximum_dist, minimum_dist, maximum_earned_money, money_per_year_per_KW ,year
    input = sorted(input)
    fitFunc = 0
    money_get = 0
    for i in range(1,len(input)):
        distance = abs(input[i] - input[i-1])
        if distance < minimum_dist:
            print(f"index:{i} isn't greater than minimum")
            print("Doesn't work out")
        elif distance <= maximum_dist:
            money_get += money_per_year_per_KW * map_value(distance, minimum_dist, maximum_dist, 1, 5)* year
        money_get += input[0]*money_per_year_per_KW * map_value(abs(input[1]-input[0]), minimum_dist, maximum_dist, 1, 5)* year
    return money_get 
   
def map_value(x, old_min, old_max, new_min, new_max):
        # 確保輸入值在舊範圍內
    x = max(min(x, old_max), old_min)
    # 變換公式
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)


def get_acc_distance(points):
    #utilize Danymic programming calculate acc_points
    accumulated_points = points.copy()
    for i in range(1, len(points)):
        accumulated_points[i][0] = accumulated_points[i-1][0] + accumulated_points[i][0]
        accumulated_points[i][1] = accumulated_points[i-1][1] + accumulated_points[i][1]
    return accumulated_points


def map_back_to_curve(straight_distances, original_points, actual_length):
    distances, acc_dist = get_straight_dist(original_points, actual_length)
    interp_x = interp1d(acc_dist, original_points[:, 0], kind='linear')
    interp_y = interp1d(acc_dist, original_points[:, 1], kind='linear')
    curve_x = interp_x(straight_distances)
    curve_y = interp_y(straight_distances)
    curve_points = np.vstack((curve_x, curve_y)).T
    return curve_points