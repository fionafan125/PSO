import os
import matplotlib.pyplot as plt
import seaborn
import pickle as pkl
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import src
import re
from src import *
'''
    "earned_money": earned_money,
    "final_ans": final_ans,
    "record_cost":fitnessCurve
'''

all_data = {
    'spend': [],
    'earned_money':[],
    'pole':[],
    'machine':[],
    'final_ans':[],
    'final_ans_2d':[]
}

def get_maximum(earned_money, machine_amount):
    earned_money, machine_amount = np.asarray(earned_money), np.asarray(machine_amount)
    idx = np.argmax(earned_money)
    max_earned_money, max_machine = earned_money[idx], machine_amount[idx]


    return max_earned_money, max_machine, idx
    

pattern_machine = r'machine_amount_(\d+)'
pattern_pole = r'wire_pole_(\d+)'
pattern_spend = r'money_([\d\.eE]+)'

main_file = os.path.join(os.path.dirname(__file__),'result_file')
file_name = 'storage_name.pkl'
#original settings

for money in os.listdir(main_file):
    money_folder = os.path.join(main_file, money)

    for wire_pole in os.listdir(money_folder):
        wire_pole_folder = os.path.join(money_folder, wire_pole)

        earned_money = []
        machine_amount = []
        final_ans_stack = []
        for machine in os.listdir(wire_pole_folder):
            read_folder = os.path.join(wire_pole_folder, machine, file_name)

            with open(read_folder, 'rb') as f:
                data = pkl.load(f)

            earned_money.append(data['earned_money'])
            final_ans_stack.append(np.asarray(data['final_ans']))

            match_machine = re.search(pattern_machine, machine)
            machine_amount_value = int(match_machine.group(1))
            machine_amount.append(machine_amount_value)

            

        max_earned_money, max_machine, idx = get_maximum(earned_money, machine_amount)
        all_data['earned_money'].append(max_earned_money)
        all_data['machine'].append(max_machine)
        all_data['final_ans'].append(final_ans_stack[idx])


        match_pole = re.search(pattern_pole, wire_pole)
        pole_amount_value = int(match_pole.group(1))
        all_data['pole'].append(pole_amount_value)

        
        match_spend = re.search(pattern_spend, money)
        spend_value = match_spend.group(1)
        all_data['spend'].append(spend_value)
        

#讀取原本excel
#原本的值mapping 回去變成點位

store_final_data = 'store_final_data.pkl'
#==settings =================================
#假設發電15年 設5瓦

store_all_folder = 'store_all'
if not os.path.exists(store_all_folder):
    os.mkdir(store_all_folder)

pkl_file_storage = os.path.join(store_all_folder, store_final_data)
record_cost = []
KW_volumn = 5.1
#===gain


meters_per_degree_latitude = 111000 #經度換算公尺

with open('point.xlsx', 'rb') as f:
    data = pd.read_excel(f)


points = np.vstack((data['經度'], data['緯度'])).T #原始座標
dist_points_meters = src.get_compared_dist(points, meters_per_degree_latitude) #相對座標
accumulated_points = src.get_acc_distance(dist_points_meters) #累計相對座標
actual_length = data['總距離'][0]

Straight_distance, accumulated_straight_dist = src.get_straight_dist(dist_points_meters, actual_length) #直線距離，累積距離





# 打印每个键对应的值
for idx in range(0, len(all_data['earned_money'])):
    point_back = src.map_back_to_curve(all_data['final_ans'][idx], points, actual_length)
    
    all_data['final_ans_2d'].append(point_back)
    print("++++++++++++++++++++++++++++++++")
    print("================================")

    for key in all_data:
        print(f"{key}: {all_data[key][idx]}")

    print("================================")
    print("++++++++++++++++++++++++++++++++")
    print("")


with open(pkl_file_storage, 'wb') as f:
    pkl.dump(all_data, f)

#===open rand=============
N = [6, 12, 18]
for wire_rand in N:
    with open(os.path.join('./random_storage', str(wire_rand) + '.pkl'), 'rb') as f:
        rand_point = pkl.load(f)
    rand_point = np.asarray(rand_point)
    rand_point = map_points_to_range(rand_point, actual_length)
    print("================================")
    print(wire_rand)
    for idx, rand_pt in enumerate(rand_point):
        if rand_pt < 16.723:
            rand_point[idx] = 16.724
    print("================================")
    rand_point2D = map_back_to_curve(rand_point, points, actual_length)
    print(rand_point2D)
    with open(os.path.join(store_all_folder, 'rand_' + str(wire_rand) + 'pkl'), 'wb') as f:
        pkl.dump(wire_rand, f)