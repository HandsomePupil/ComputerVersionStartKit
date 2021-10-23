from functools import total_ordering
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("machine_learning/data.csv", delimiter=",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()

lr = 0.0001
b = 0
k = 0
epochs = 50

def compute_error(b,k,x_data,y_data):
    totalError = 0
    for i in range(len(x_data)):
        totalError = totalError + (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / (2.0 * float(len(x_data)))

def gradient_descent(x_data, y_data, b, k, lr, epochs):
    m = float(len(x_data))

    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        for j in range(0, len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)

        if i % 5 == 0:
            print("Epochs:",i)
            plt.plot(x_data, y_data, 'b.')    #格式控制字符串:"颜色", "点型", "线型";"b"为蓝色, "o"为圆点, ":"为点线
            plt.plot(x_data, k*x_data + b, 'r')
            plt.show()
    return b, k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))
