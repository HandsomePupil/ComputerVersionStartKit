from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# 真实函数y = sin(2πx)
def real_func(x):
    return np.sin(2*np.pi*x)

# 回归函数，p是参数项
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差计算
def residual_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 构造10个样本点以及1000个绘图点
x = linspace(0, 1, 10)
x_points = linspace(0, 1, 1000)
# 正态分布噪音值构建
y = [np.random.normal(0, 0.1)+i for i in real_func(x)]

def fitting(M=0): #此处M相当于定义了回归预测函数的参数个数
    """
    n 为 多项式的次数
    """    
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residual_func, p_init, args=(x, y))
    # leastsq()中三个输入分别为1.CostFunction 2.函数的参数 3.数据点坐标
    # 该函数的返回值有1.拟合后的参数数组 2.函数运行Info（0-9的数字编码）
    print('Fitting Parameters:', p_lsq[0])  #该出取p_lsq返回值中的1即可，见上一个备注

    # Visilization
    plt.plot(x_points, real_func(x_points), label = "real")
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()  #用于显示label等图示信息
    plt.show()
    return p_lsq

# RUN
p_lsq_0 = fitting(M=0)
p_lsq_1 = fitting(M=1)
p_lsq_2 = fitting(M=2)
p_lsq_8 = fitting(M=8)
