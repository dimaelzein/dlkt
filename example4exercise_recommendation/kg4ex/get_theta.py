import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# 定义遗忘曲线函数
def forgetting_curve(t, lambda_):
    return np.exp(-lambda_ * t)


# 给定数据点
times = np.array([1 / 3, 1, 9, 24, 48])  # 时间转换为小时
retentions = np.array([0.582, 0.442, 0.358, 0.337, 0.278])  # 记忆保持率

# 进行曲线拟合
popt, pcov = curve_fit(forgetting_curve, times, retentions)

# 提取拟合参数
lambda_opt = popt[0]

# 打印拟合参数
print(f"最优遗忘率参数 λ: {lambda_opt}")

# 绘制拟合曲线和原始数据点
t_fit = np.linspace(0, 50, 100)  # 生成拟合曲线的时间点
r_fit = forgetting_curve(t_fit, lambda_opt)  # 计算拟合曲线的记忆保持率

plt.scatter(times, retentions, color='red', label='Data Points')  # 绘制数据点
plt.plot(t_fit, r_fit, label='Fitted Curve')  # 绘制拟合曲线
plt.xlabel('Time (hours)')
plt.ylabel('Retention Rate')
plt.legend()
plt.show()
