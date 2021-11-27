import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from displayData import displayData
from displayData import displayImage

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def forwardProcess(x, num_capas, thetas):
    a = x
    for i in range(num_capas):
        aNew = np.hstack([np.ones([x.shape[0], 1]), a])
        a = sigmoide(np.dot(aNew,thetas[i].T))

    return a

def coste(x, y_ones, num_capas, thetas):
    res = forwardProcess(x, num_capas, thetas)
    
    aux = -(y_ones) * np.log(res)

    aux2 = (1 - y_ones) * np.log(1-res)

    #print((aux - aux2))

    return np.sum(aux - aux2) / x.shape[0]


def parte1():
    data = loadmat("Data/ex4data1.mat")

    x = data['X']
    y = data['y']
    yR = np.ravel(y)


    m = np.shape(x)[0]
    n = np.shape(x)[1]

    numExamples = 100
    numCapas = 2
    numLabels = 10

    #print(yR[700])

    yR[yR == 10] = 0
    #yR = (yR - 1)
    y_onehot = np.zeros((m, numLabels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][yR[i]] = 1

    #print(np.shape(x))
    #print(y_onehot.shape)

    pesos = loadmat("Data/ex4weights.mat")

    theta1, theta2 = pesos['Theta1'], pesos['Theta2']

    thetas = np.array([theta1, theta2], dtype='object')

    #print(np.shape(theta1))
    #print(np.shape(theta2))
    
    #cost = coste(x[:1, :], y_onehot[:1, :], numCapas, thetas)
    cost = coste(x, y_onehot, numCapas, thetas)

    print(cost)

    sample = np.random.choice(m, numExamples)

    #displayData(x[sample, :])

    #displayImage(x[700, :])

    #plt.show()
"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte2():
    print("aun no llegamos")

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte3():
    print("aun no llegamos")


def main():
    parte1()
    #parte2()
    #parte3()

if __name__ == "__main__":
    main()