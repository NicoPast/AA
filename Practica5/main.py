import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

def costeRegul(x, y, thetas, reg):
    m = np.shape(x)[0]
    hTh = np.dot(x,thetas)
    return (np.sum((hTh - y.T) ** 2)) / (2*m) + (reg / (2*m)) * np.sum(thetas[1:] ** 2)

def gradiente(x, y, thetas):
    return (np.dot((np.dot(x, thetas) - y.T), x)) / x.shape[0]

def gradienteRegul(x, y, thetas, reg):
    grad = gradiente(x, y, thetas)

    regul = np.sum(thetas[1:] * (reg / x.shape[0]))

    return grad + regul

def parte1():
    data = loadmat("Data/ex5data1.mat")

    x = data['X']
    y = data['y']
    yR = np.ravel(y)

    m = np.shape(x)[0]
    n = np.shape(x)[1]

    xNew = np.hstack([np.ones([m, 1]), x])

    thetas = np.ones(n+1)
    
    grad = gradienteRegul(xNew, yR, thetas, 1)
    print(grad)

    minG = np.min(grad)
    maxG = np.max(grad)

    cost = costeRegul(xNew, yR, thetas, 1)   
    print(cost)

    xVal = data['Xval']
    yVal = data['yval']

    xTest = data['Xtest']
    yTest = data['ytest']

    plt.scatter(x, y, color='red', marker='x')
    #plt.plot(x, y, "x", color='red')
    plt.show()
    
def main():
    parte1()

if __name__ == "__main__":
    main()