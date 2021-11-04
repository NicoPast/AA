import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from scipy.io import loadmat

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def costeReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return -((np.dot(np.log(h), y) + np.dot(np.log(1 - h + 1e-6), 1 - y)) / len(x)) + (l/(2*len(x))) * l * np.sum(thetas[1:] ** 2)

def gradienteReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(y) + (thetas * l) / len(y)

def oneVsAll(x, y, num_etiquetas, reg):
    theta = np.zeros((num_etiquetas, x.shape[1])) 

    for i in range(num_etiquetas):
        theta[i] = opt.fmin_tnc(func=costeReg, x0=theta[i], fprime=gradienteReg, args=(x,(y==(i+9)%10+1)*1,reg))[0]

    return theta

def parte1():
    data = loadmat("Data/ex3data1.mat")

    x = data['X']
    y = data['y']
    yR = np.ravel(y)

    m = np.shape(x)[0]
    n = np.shape(x)[1]

    numClases = 10
    numExamples = 20

    reg = 1.0

    xNew = np.hstack([np.ones([m, 1]), x])

    theta = oneVsAll(xNew, yR, numClases, reg)

    sample = np.random.choice(xNew.shape[0], numExamples)

    correct = 0
    for i in range(m):
        result = sigmoide(np.matmul(theta, xNew[i, :]))
        id = np.argmax(result)
        if(y[i][0]%10 == (id+1)%10):
            correct+= 1

    correct = correct/m

    print("Correct values are:", (1-correct)*100, "%")
    
    for i in range(numExamples):
        result = sigmoide(np.matmul(theta, xNew[sample, :][i]))
        max = np.max(result)
        id = np.argmax(result)

        print("sample",i,y[sample,:][i]%10,": creemos que es un", id, "con una certeza del", max)

    plt.imshow(x[sample, :].reshape(-1, 20).T)
    plt.axis('off')

    plt.show()

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def forwardProcess(x, num_capas, thetas):
    a = x
    for i in range(num_capas):
        aNew = np.hstack([np.ones([x.shape[0], 1]), a])
        a = sigmoide(np.dot(aNew,thetas[i].T))

    return a

def parte2():
    data = loadmat("Data/ex3data1.mat")

    x = data['X']
    y = data['y']
    yR = np.ravel(y)

    m = np.shape(x)[0]
    n = np.shape(x)[1]

    numExamples = 20

    pesos = loadmat("Data/ex3weights.mat")

    theta1, theta2 = pesos['Theta1'], pesos['Theta2']

    thetas = np.array([theta1, theta2], dtype='object')

    sample = np.random.choice(m, 1)

    res = forwardProcess(x, 2, thetas)

    sample = np.random.choice(m, numExamples)
    
    correct = 0
    for i in range(m):
        id = np.argmax(res[i, :])
        if(y[i][0]%10 == (id+1)%10):
            correct+= 1

    correct = correct/m

    print("Correct values are:", correct*100, "%")

    for i in range(numExamples):
        max = np.max(res[sample, :][i])
        id = np.argmax(res[sample, :][i])

        print("sample", i, y[sample,:][i]%10, ": creemos que es un", (id+1)%10, "con una certeza del", max)
    
    plt.imshow(x[sample, :].reshape(-1, 20).T)
    plt.axis('off')

    plt.show()
    
if __name__ == "__main__":
    parte1()
    parte2()