'''畫year的摺線圖'''

import os
import matplotlib.pyplot as plt
import seaborn
import pickle as pkl
import numpy as np
from matplotlib.ticker import MaxNLocator
import re
'''
store = {
    "earned_money": earned money
    "final_ans": position
    "record_cost": record_cost
}
'''

money_spend = ['1e7', '1.5e7', '2e7']
pole = [6, 12, 18]
storage_main_file = './machine_png'
main_file = './result_file'

if not os.path.exists(storage_main_file):
    os.makedirs(storage_main_file)

storage_main_file = os.path.join(storage_main_file, )
if not os.path.exists(storage_main_file):
    os.makedirs(storage_main_file)

pkl_file = 'storage_name.pkl'


all_data = []

def plot_fig(machine_amounts, data, money, pole):

    pattern = r'wire_pole_(\d+)'
    match = re.search(pattern, pole)
    pole_value = int(match.group(1))

    money_folder = os.path.join(storage_main_file, money)
    if not os.path.exists(money_folder):
        os.mkdir(money_folder)

    pole_folder = os.path.join(money_folder, pole)
    if not os.path.exists(pole_folder):
        os.mkdir(pole_folder)
    

    plt_png = os.path.join(pole_folder, str(machine_amounts) + '.png')
    # print(machine_amounts.shape)
    machine_draw = range(2, machine_amounts+2)
    print(machine_draw)
    print(data.shape)
    plt.plot(machine_draw, data, marker='o', linestyle='-', color='b', label='Label of Line')  # 繪製折線圖，設置標記、線型、顏色和標籤
    plt.title('Spend ' + money + '; pole ' + str(pole_value))  # 設置圖形標題
    plt.xlabel('machine amount')  # 設置x軸標籤
    plt.ylabel('Earned money')  # 設置y軸標籤
    ax = plt.gca()  # 获取当前轴
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置x轴标签为整数
    plt.grid(True)  # 顯示網格
    plt.savefig(plt_png)  # 顯示圖形
    plt.close()
    print("finish drawing")
    print(plt_png)

for money in money_spend:
    money_file = 'money_' + money
    for wire_pole in pole:
        wire_pole_file = 'wire_pole_' + str(wire_pole)
        read_folder = os.path.join(main_file, money_file, wire_pole_file)
        print(read_folder)
        
        mc_storage = []
        for machine_folder in os.listdir(read_folder):  
            read_file = os.path.join(read_folder, machine_folder, pkl_file)
            with open(read_file, 'rb') as f:
                data = pkl.load(f)
            earned_money = data['earned_money']
            mc_storage.append(earned_money)
        
        
        plot_fig(len(os.listdir(read_folder)), np.asarray(mc_storage), money, wire_pole_file)