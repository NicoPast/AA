import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from checkNNGradients import checkNNGradients
from displayData import displayData
from displayData import displayImage

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def forwardProp(x, num_capas, thetas):
    a = x
    for i in range(num_capas):
        aNew = np.hstack([np.ones([x.shape[0], 1]), a])
        a = sigmoide(np.dot(aNew,thetas[i].T))

    return a

def coste(x, y_ones, num_capas, thetas):
    res = forwardProp(x, num_capas, thetas)
    
    aux = -(y_ones) * np.log(res)

    aux2 = (1 - y_ones) * np.log(1-res)

    return np.sum(aux - aux2) / x.shape[0]

def costeRegul(x, y_ones, num_capas, thetas, l):

    cost = coste(x, y_ones, num_capas, thetas)

    vals = np.zeros(num_capas)

    for i in range(num_capas):
        vals[i] = np.sum(thetas[i] ** 2) - np.sum(thetas[i][:, 0] ** 2)

    regul = np.sum(vals) * l / (2*x.shape[0]) 

    return cost + regul

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

    yR = (yR - 1)
    y_onehot = np.zeros((m, numLabels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][yR[i]] = 1

    pesos = loadmat("Data/ex4weights.mat")

    # red neuronal 400 neuronas input
    # 25 hidden
    # 10 output
    theta1, theta2 = pesos['Theta1'], pesos['Theta2']

    thetas = np.array([theta1, theta2], dtype='object')

    cost = costeRegul(x, y_onehot, numCapas, thetas, 1)

    print(cost)

    #sample = np.random.choice(m, numExamples)

    #displayData(x[sample, :])

    #displayImage(x[700, :])

    #plt.show()

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, x, y, reg):
    # backprop devuelve una tupla (coste, gradiente) con el coste y el gradiente de
    # una red neuronal de tres capas, con num_entradas, num_ocultas nodos en la capa
    # oculta y num_etiquetas nodos en la capa de salida. Si m es el numero de ejemplos
    # de entrenamiento, la dimension de 'X' es (m, num_entradas) y la de 'y'' es
    # (m, num_etiquetas)

    if(num_ocultas.shape[0] + 2 < 3):
        print("ERROR: num_capas incorrect, must have an input, at least one hidden and an output layer")
        return (0,0)

    # calculo de thetas
    thetas = np.empty(num_ocultas.shape[0] + 1, dtype='object')
    pointer = num_ocultas[0] * (num_entradas + 1)

    thetas[0] = np.reshape(params_rn[: pointer] , (num_ocultas[0], (num_entradas + 1)))

    for i in range(1, num_ocultas.shape[0]):
        thetas[i] = np.reshape(params_rn[pointer : pointer + num_ocultas[i] * (num_ocultas[i-1] + 1)], (num_ocultas[i], (num_ocultas[i-1] + 1)))
        pointer += num_ocultas[i] * (num_ocultas[i-1] + 1)
        
    thetas[num_ocultas.shape[0]] = np.reshape(params_rn[pointer :] , (num_etiquetas, (num_ocultas[-1] + 1)))

    # theta1 = np.reshape(params_rn[: num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    # theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1) :], (num_etiquetas, (num_ocultas + 1)))

    # thetas = np.array([theta1, theta2], dtype='object')

    hTheta = forwardProp(x, thetas.shape[0], thetas)

    #print(hTheta.shape)

    delta3 = hTheta - y

    #test = a2 *

    delta2 = np.dot(delta3, thetas[1])

    print(delta2.shape)

    # print(thetas.shape)

    # print(thetas[0].shape)
    # print(thetas[1].shape)

    cost = costeRegul(x, y, thetas.shape[0], thetas, 1)

    return (cost,0)


def parte2():
    print(checkNNGradients(backprop, 1))

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte3():
    print("aun no llegamos")


def main():
    #parte1()
    parte2()
    #parte3()

if __name__ == "__main__":
    main()