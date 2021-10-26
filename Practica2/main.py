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

def pinta_frontera_lineal(theta, x, y):
    plt.figure()

    zoom = 5

    x1min, x1max = x[:,0].min(),x[:,0].max()
    x2min, x2max = x[:,1].min(),x[:,1].max()

    xx1,xx2 = np.meshgrid(np.linspace(x1min,x1max),np.linspace(x2min,x2max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))

    print(h)
    h = h.reshape(xx1.shape)

    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where(y==1)
    # Dibuja los ejemplos positivos
    plt.axis([x1min-zoom, x1max+zoom, x2min-zoom, x2max+zoom])
    plt.scatter(x[pos,0], x[pos, 1], marker='+', label="Admitted")
    pos = np.where(y==0)
    plt.scatter(x[pos,0], x[pos, 1], label="Not admitted")

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    plt.legend(loc="upper right")
    plt.show()

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

    # x1min = np.min(x[:,0])
    # x1max = np.max(x[:,0])
    # x2min = -(theta_opt[0] + (x1min * theta_opt[1])) / theta_opt[2]
    # x2max = -(theta_opt[0] + (x1max * theta_opt[1])) / theta_opt[2]

    # plt.figure()
    # # Obtiene un vector con los índices de los ejemplos positivos
    # pos = np.where(y==1)
    # # Dibuja los ejemplos positivos
    # plt.scatter(x[pos,0], x[pos, 1], marker='+', label="Admitted")
    # pos = np.where(y==0)
    # plt.scatter(x[pos,0], x[pos, 1], label="Not admitted")
    # plt.plot([x1min,x2min], [x1max, x2max])
    # plt.legend(loc="upper right")
    # plt.show()

    pinta_frontera_lineal(theta_opt, x, y)

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""
def costeReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return -((np.dot(np.log(h), y) + np.dot(np.log(1 - h), 1 - y)) / len(x)) + (l/(2*len(x))) * l * np.sum(thetas[1:] ** 2)

def gradienteReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(y) + (thetas * l) / len(y)

def pinta_frontera_poli(thetas, x, y, poly):
    plt.figure()

    zoom = 0.1

    x1min, x1max = x[:, 0].min(), x[:, 0].max()
    x2min, x2max = x[:, 1].min(), x[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1min, x1max), np.linspace(x2min, x2max))

    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(thetas))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where(y==1)
    # Dibuja los ejemplos positivos
    plt.scatter(x[pos,0], x[pos, 1], marker='+', label="Admitted")
    pos = np.where(y==0)
    plt.scatter(x[pos,0], x[pos, 1], label="Not admitted")
    plt.legend(loc="upper right")
    plt.axis([x1min-zoom, x1max+zoom, x2min-zoom, x2max+zoom])
    plt.show()

def parte2():
    data = loadCSV("Data\ex2data2.csv")
    x = data[:, :-1]
    y = data[:, -1]

    l = 1.0

    polynomial = sk.PolynomialFeatures(6)
    xNew = polynomial.fit_transform(x)

    n = np.shape(xNew)[1]
    thetas = np.zeros(n)

    print(costeReg(thetas, xNew, y, l))
    print(gradienteReg(thetas, xNew, y, l))

    result = opt.fmin_tnc(func=costeReg, x0=thetas, fprime=gradienteReg, args=(xNew,y,l))
    theta_opt = result[0]
    print(coste(theta_opt,xNew,y))

    pinta_frontera_poli(theta_opt, x, y, polynomial)
    
if __name__ == "__main__":
    parte1()
    parte2()