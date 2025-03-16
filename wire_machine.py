import subprocess
import numpy as np
# 定义 Python 解释器和脚本名称


python_executable = "python"
script_name = "new_pso.py"
cost = 380000
spend_money = [1e7, 1.5e7, 2e7]
machine_amount_stack = np.asarray([int(money // cost) for money in spend_money])
# print(machine_amount_stack)
spend_money = ['1e7', '1.5e7', '2e7']
print(spend_money)
# 假设 machine_amount 的范围是 1 到 5，wire_pole 的數輛是10, 15, 20
wire_pole = [6, 12, 18]



# 使用两个嵌套 for 循环来遍历所有 machine_amount 和 wire_pole 的组合
for idx, machine_amount in enumerate(machine_amount_stack):
    for i in range(2, machine_amount):
        for j in wire_pole:
            # 构建命令和参数
            command = [python_executable, script_name, "--machine_amount", str(i), "--wire_pole", str(j), "--money_spend", str(spend_money[idx])]
            try:
                # 执行脚本
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print(f"运行成功: machine_amount={i}, wire_pole={j}")
                print("脚本输出:", result.stdout)  # 打印标准输出
            except subprocess.CalledProcessError as e:
                print(f"错误发生在: machine_amount={i}, wire_pole={j}")
                print("错误输出:", e.stderr)  # 打印标准错误输出

