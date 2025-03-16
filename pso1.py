import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os
import argparse
import pandas as pd
import pickle as pkl
from math import cos, radians

main_file = os.path.join(os.path.dirname(__file__))
store_png_file = "png_file"
penalty_base = 1e5 #改這個
store_result_file = 'result_file'

if not os.path.exists(store_result_file):
    os.makedirs(store_result_file)

storage_name = 'storage_name.pkl'

#==settings =================================
#假設發電15年 設5瓦
parser = argparse.ArgumentParser(description="加參數")
parser.add_argument('--machine_amount', type=int, help="機器數量實驗")
parser.add_argument('--wire_pole', type=int, help="random type")
parser.add_argument('--money_spend', type=str, help="record number")
args = parser.parse_args()

tmp = os.path.join(main_file, store_result_file, 'money_' + str(args.money_spend))
if not os.path.exists(tmp):
    os.mkdir(tmp)

store_file = os.path.join(tmp ,'wire_pole_' + str(args.wire_pole))

if not os.path.exists(store_file):
    os.makedirs(store_file)

store_file = os.path.join(store_file,'machine_amount_'+ str(args.machine_amount) )
print(store_file)
if not os.path.exists(store_file):
    os.makedirs(store_file)

pkl_file = os.path.join(store_file, storage_name)
png_file = os.path.join(store_file, 'converge_curve.png')
year = 10 #假設

iterations = 200  # 迭代次數
record_cost = []
maximum_dist = 50
minimum_dist = 40
KW_volumn = 5.1
#===gain
money_per_year_per_KW = 38551.68 #錢/年1KW
machine_amount = args.machine_amount
maximum_GetBack_money = money_per_year_per_KW * KW_volumn * year * machine_amount
#===cost
cost = 380000
cost_money = cost * KW_volumn * machine_amount
maximum_earned_money = maximum_GetBack_money - cost_money

if maximum_GetBack_money < cost_money:
    print("QAQ")
    exit()

Dim = 1         # dimension
pop_size = 20   # population_size
w_it_max = 0.9  # inertia weight max 
w_it_min = 0.4  # inertia weight min 
c1 = 2          # acceleration factor
c2 = 2          # acceleration factor
max_iter = 1900 # iterations
Lb = -5.12 * np.ones(Dim)  # lower bound
Ub = 5.12 * np.ones(Dim)   # upper bound

# Rastrigin function 1D
def fun(X):
    x1 = X[:,0]
    obj_val = 20 + (x1**2 - 10 * np.cos(2 * np.pi * x1))
    return obj_val

# Initialize vel and position for PSO
pos = np.zeros((pop_size, Dim))
vel = np.zeros((pop_size, Dim))
for i in range(pop_size):
    for j in range(Dim):
        pos[i,j] = Lb[j] + (Ub[j] - Lb[j]) * np.random.random()
for i in range(pop_size):
    for j in range(Dim):
        vel[i,j] = np.random.random()

# Calculate target value         
obj_val = fun(pos)

# Record pbest
pbestval = obj_val.copy()
pbest = pos.copy()
    
# Record gbest
fminval = np.min(obj_val)
index = np.argmin(obj_val)
gbest = pbest[index]

record = []
# PSO Iteration
for iter in range(max_iter):     
    # Calculate target value 
    obj_val = fun(pos)
       
    # Update pbest
    p = np.where(obj_val < pbestval)
    pbest[p] = pos[p] 
    pbestval[p] = obj_val[p]
    
    # Update gbest
    fbestval = np.min(pbestval)
    ind = np.argmin(pbestval)
    record.append(fbestval)
    
    if fbestval <= fminval:
        fminvalue = fbestval
        gbest = pbest[ind]
    
    # Update vel and position
    w = w_it_max - (iter / max_iter) * (w_it_max - w_it_min)
    
    for i in range(pop_size):
        for j in range(Dim):
            vel[i,j] = w * vel[i,j] + c1 * np.random.random() * (pbest[i,j] - pos[i,j]) + \
                       c2 * np.random.random() * (gbest[j] - pos[i,j])        
            pos[i,j] = pos[i,j] + vel[i,j]
                
            # Check bound
            if pos[i,j] < Lb[j]:
                pos[i,j] = Lb[j]
            elif pos[i,j] > Ub[j]:
                pos[i,j] = Ub[j]

print("最佳解:", gbest)
print("最小值:", fbestval)

fig = plt.figure() 
plt.plot([i for i in range(max_iter)], record)
plt.xlabel("iteration") 
plt.ylabel("object value")
plt.show()

X = np.linspace(-5.12, 5.12, 100)
Y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(X, Y)
Z = 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="nipy_spectral")
ax.contourf(X, Y, Z, zdir='z', offset=-2)
plt.title("Rastrigin Function")
plt.show()

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

def compute_dis_xy(init_pos, final_pos, meters_per_degree_latitude):
    # 計算南北方向上的距離（Δ緯度 × 每度緯度的公尺數）
    delta_latitude = final_pos[1] - init_pos[1]
    distance_north_south = delta_latitude * meters_per_degree_latitude

    # 計算東西方向上的距離（Δ經度 × cos(平均緯度) × 每度緯度的公尺數）
    average_latitude = radians((init_pos[1] + final_pos[1]) / 2)
    delta_longitude = final_pos[0] - init_pos[0]
    distance_east_west = delta_longitude * cos(average_latitude) * meters_per_degree_latitude
    return [distance_north_south, (-1)*distance_east_west]

def get_compared_dist(points, meters_per_degree_latitude):
    compared_dist = []
    for point in points:
        tmp_dist = compute_dis_xy(points[0,:], point, meters_per_degree_latitude)
        compared_dist.append(tmp_dist)
    compared_dist = np.array(compared_dist)
    return compared_dist

# useless function
def get_acc_distance(points):
    # utilize Danymic programming calculate acc_points
    accumulated_points = points.copy()
    for i in range(1, len(points)):
        accumulated_points[i][0] = accumulated_points[i-1][0] + accumulated_points[i][0]
        accumulated_points[i][1] = accumulated_points[i-1][1] + accumulated_points[i][1]
    return accumulated_points

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
    global length, maximum_dist, minimum_dist, maximum_earned_money, money_per_year_per_KW ,year, wire_pole
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
    for point in wire_pole:
        smallest_dist = 0
        for search_point in input:
            smallest_dist = min(abs(search_point - point), smallest_dist)
        fitFunc -= smallest_dist * ((smallest_dist//100)/10 + 1) #<100 =>1 ； > 100 * 階梯式
    return money_get 

def map_value(x, old_min, old_max, new_min, new_max):
    # 確保輸入值在舊範圍內
    x = max(min(x, old_max), old_min)
    # 變換公式
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)

meters_per_degree_latitude = 111000 #經度換算公尺

with open('point.xlsx', 'rb') as f:
    data = pd.read_excel(f)

points = np.vstack((data['經度'], data['緯度'])).T #原始座標
dist_points_meters = get_compared_dist(points, meters_per_degree_latitude) #相對座標
accumulated_points = get_acc_distance(dist_points_meters) #累計相對座標
actual_length = data['總距離'][0]

Straight_distance, accumulated_straight_dist = get_straight_dist(dist_points_meters, actual_length) #直線距離，累積距離

#================================read wire_pole================================
with open(os.path.join(main_file,'random_storage', str(args.wire_pole)+'.pkl'), 'rb') as f:
    rand_pole = pkl.load(f)

wire_pole = map_points_to_range(rand_pole, actual_length)

#取出起始點[0,0] 終點
init_pos = accumulated_straight_dist[0]
final_dist = accumulated_straight_dist[-1]

# #get distance
length = np.linalg.norm(final_dist - init_pos) 
init_pos = 0
final_dist = length

print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('your cost is:', cost_money)
print('distance :', length)
print('machine amount :', machine_amount)
print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# # run

#main function
X = sampleGeneartor(length)
LB = np.asarray([0]*machine_amount)
UB = np.asarray([length]*machine_amount)

fitnessCurve, para = pso(X, dim=machine_amount, whale_num=60, max_iter=2000, LB = LB, UB = UB, b = 2).opt()

final_ans = sorted(para)
print(final_ans)
earned_money = earned_money_count(final_ans) - cost_money
print(f"earned moeny is: {earned_money}")

plot_converge_curve(fitnessCurve, storage_name = png_file) 
store = {
    "earned_money": earned_money,
    "final_ans": final_ans,
    "record_cost":fitnessCurve
}

with open(pkl_file, 'wb') as f:
    pkl.dump(store, f)
