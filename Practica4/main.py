import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

from scipy.io import loadmat

from checkNNGradients import checkNNGradients
from displayData import displayData
from displayData import displayImage

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def forwardProp(x, num_capas, thetas):
    a = np.empty(num_capas + 1, dtype="object")
    a[0] = x
    for i in range(num_capas):
        aNew = np.hstack([np.ones([x.shape[0], 1]), a[i]])
        a[i] = aNew
        a[i+1] = sigmoide(np.dot(aNew,thetas[i].T))

    return a

def coste(x, y_ones, num_capas, thetas):
    res = forwardProp(x, num_capas, thetas)[num_capas]

    return np.sum((-(y_ones) * np.log(res)) - ((1 - y_ones) * np.log(1-res))) / x.shape[0]

def costeRegul(x, y_ones, num_capas, thetas, reg):
    cost = coste(x, y_ones, num_capas, thetas)

    val = 0

    for i in range(num_capas):
        val += np.sum(np.power(thetas[i][1:], 2))

    regul = val * (reg / (2*x.shape[0]))

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

def lineal_back_prop(x, y, thetas, reg):
    m = x.shape[0]
    Delta1 = np.zeros_like(thetas[0])
    Delta2 = np.zeros_like(thetas[1])

    for t in range(m):
        hThetaTot = forwardProp(x, thetas.shape[0], thetas)
        a1t = hThetaTot[0][t, :] # (401,)
        a2t = hThetaTot[1][t, :] # (26,)
        ht = hThetaTot[2][t, :] # (10,)
        yt = y[t] # (10,)

        d3t = ht - yt # (10,)
        d2t = np.dot(thetas[1].T, d3t) * (a2t * (1 - a2t)) # (26,)

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    gradient1 = Delta1 / m
    gradient2 = Delta2 / m
    
    col0 = gradient1[0]
    gradient1 = gradient1 + (reg/m)*thetas[0]
    gradient1[0] = col0

    col0 = gradient2[0]
    gradient2 = gradient2 + (reg/m)*thetas[1]
    gradient2[0] = col0

    return np.append(gradient1, gradient2).reshape(-1)

def vect_back_prop(x, y, thetas, reg):
    m = x.shape[0]

    hThetaTot = forwardProp(x, thetas.shape[0], thetas)

    dlts = np.empty_like(thetas)
    Deltas = np.empty_like(thetas)

    dlts[-1] = hThetaTot[-1] - y

    for i in range(1, thetas.shape[0]):
        a = hThetaTot[-(i+1)]

        delta = np.dot(thetas[-i].T, dlts[-i].T).T

        delta = delta * a * (1-a)

        delta = delta[:,1:]
        
        dlts[-(i+1)] = delta

    res = []

    for i in range(thetas.shape[0]):
        Deltas[i] = np.dot(dlts[i].T, hThetaTot[i]) / m
        Deltas[i] = np.append(Deltas[i][0], Deltas[i][1:] + (reg/m) * thetas[i][1:])
        res = np.append(res, Deltas[i]).reshape(-1)

    return res

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

    # m = x.shape[0]

    # hThetaTot = forwardProp(x, thetas.shape[0], thetas)

    # delta3 = hThetaTot[-1] - y

    # a2 = hThetaTot[-2]

    # delta2 = np.dot(thetas[1].T, delta3.T).T

    # delta2 = delta2 * a2 * (1-a2)

    # delta2 = delta2[:,1:]

    # Delta1M = np.dot(delta2.T, hThetaTot[0]) / m
    # Delta2M = np.dot(delta3.T, hThetaTot[1]) / m

    # Delta1M = np.append(Delta1M[0], Delta1M[1:] + (reg/m) * thetas[0][1:])
    # Delta2M = np.append(Delta2M[0], Delta2M[1:] + (reg/m) * thetas[1][1:])

    #lineal_back_prop(x, thetas, reg)

    return costeRegul(x, y, thetas.shape[0], thetas, reg), vect_back_prop(x, y, thetas, reg)


def parte2():
    print(np.sum(checkNNGradients(backprop, 1)))

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