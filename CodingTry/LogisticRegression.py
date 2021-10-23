from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
# scale = True
#数据载入与预处理
data = np.genfromtxt("machine_learning/LR-testSet.csv", delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]

def plot():
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(len(x_data)):
        if(y_data[i]==0):
            x0.append(x_data[i,0])
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])
    scatter0 = plt.scatter(x0,y0,c='b',marker='o')
    scatter1 = plt.scatter(x1,y1,c='r',marker='x')
    plt.legend(handles=[scatter0,scatter1],labels=['label0','label1'],loc='best')
plot()
plt.show()