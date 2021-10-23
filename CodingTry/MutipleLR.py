from functools import total_ordering
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("machine_learning/资料/程序/回归/Delivery.csv",delimiter=',')

x_data = data[:,0:2]   # data[x:y, m:n] 取数据行列，大于等于x，小于y
y_data = data[:,2]

# HyperParameters Setting
lr = 0.0001
theta0 = 0
theta1 = 0
theta2 = 0
epochs = 50

def compute_error(theta0, theta1, theta2, x_data, y_data):
    totalError = 0
    for i in range(len(x_data)):
        totalError += (y_data[i] - (theta1 * x_data[i,0] + theta2 * x_data[i,1] + theta0))
    return totalError / float(len(x_data))

def gradient_descent(x_data, y_data, theta0, theta1, theta2, lr, epochs):
    m = float(len(x_data))
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0, len(x_data)):
            theta0_grad += (1/m)*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            theta1_grad += (1/m)*(x_data[j,0])*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            theta2_grad += (1/m)*(x_data[j,1])*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
        #update
        theta0 -= lr * theta0_grad
        theta1 -= lr * theta1_grad
        theta2 -= lr * theta2_grad
        #运行中输出参数变化
        print("epochs:{0},error:{1}".format(i,compute_error(theta0,theta1,theta2,x_data,y_data)))
    return theta0, theta1, theta2

print("Starting theta0:{0}, theta1:{1}, theta2:{2}, error:{3}".
        format(theta0,theta1,theta2,compute_error(theta0,theta1,theta2,x_data,y_data)))
print("Running...")
theta0, theta1, theta2 = gradient_descent(x_data, y_data, theta0, theta1, theta2, lr, epochs)
print("After {0} epochs , theta0 = {1}, theta1 = {2}, theta2 = {3}, error = {4}".
        format(epochs,theta0,theta1,theta2,compute_error(theta0,theta1,theta2,x_data,y_data)))

ax = plt.figure().add_subplot(111, projection = '3d') 
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 100) #点为红色三角形  
x0 = x_data[:,0]
x1 = x_data[:,1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = theta0 + x0*theta1 + x1*theta2
# 画3D图
ax.plot_surface(x0, x1, z)
#设置坐标轴  
ax.set_xlabel('Miles')  
ax.set_ylabel('Num of Deliveries')  
ax.set_zlabel('Time')  
  
#显示图像  
plt.show()  