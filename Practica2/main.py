import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
import scipy as scpy
import sklearn.preprocessing as sk

def loadCSV(fileName):
    return read_csv(fileName, header=None).to_numpy().astype(float)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def coste(thetas, x, y):
    h = sigmoide(np.dot(x, thetas))
    return -1/len(x) *(np.dot(np.log(h), y) + np.dot(np.log(1-h), 1-y))

def gradiente(thetas, x, y):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(x)

def parte1():
    data = loadCSV("Data/ex2data1.csv")

    x = data[:, :-1]
    y = data[:, -1]

    m = np.shape(x)[0]
    n = np.shape(x)[1]

    xNew = np.hstack([np.ones([m, 1]), x])

    thetas = np.zeros(n+1)
    # print(coste(thetas, xNew, y))
    # print(gradiente(thetas, xNew, y))

    result = opt.fmin_tnc(func=coste, x0=thetas, fprime=gradiente, args=(xNew,y))
    theta_opt = result[0]
    print(coste(theta_opt,xNew,y))

    x1min = np.min(x[:,0])
    x1max = np.max(x[:,0])
    x2min = -(theta_opt[0] + (x1min * theta_opt[1])) / theta_opt[2]
    x2max = -(theta_opt[0] + (x1max * theta_opt[1])) / theta_opt[2]

    plt.figure()

    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where(y==1)
    # Dibuja los ejemplos positivos
    plt.scatter(x[pos,0], x[pos, 1], marker='+', label="Admitted")
    pos = np.where(y==0)
    plt.scatter(x[pos,0], x[pos, 1], label="Not admitted")
    plt.plot([x1min,x2min], [x1max, x2max])
    plt.legend(loc="upper right")
    plt.show()

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""
def costeReg(thetas, x, y, l):
    # TF is lambda?
    h = sigmoide(np.dot(x, thetas))
    return - ((np.dot(np.log(h), y) + np.dot(np.log(1 - h), 1 - y)) / len(x)) + (l/(2*len(x)) ) #falta el sumatorio HALP

def gradienteReg():
    print("AAAA")

def parte2():
    data = loadCSV("Practica2\Data\ex2data2.csv")
    x = data[:, :-1]
    y = data[:, -1]

    n = np.shape(x)[1]

    thetas = np.zeros(n+1)
    l = 1

    polynomial = sk.PolynomialFeatures(6)
    X  = polynomial.fit_transform(x)

    print(costeReg(thetas,x, y, l))
    




if __name__ == "__main__":
    #parte1()
    parte2()