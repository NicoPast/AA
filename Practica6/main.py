import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

import sklearn.svm as svm

from sklearn.metrics import accuracy_score

def scatter_mat(x,y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(x[pos, 0], x[pos, 1], color='black', marker='+')
    plt.scatter(
    x[neg, 0], x[neg, 1], color='yellow', edgecolors='black', marker='o')

def visualize_boundary(x, y, svm, zoom):
    x1 = np.linspace(x[:, 0].min() - zoom, x[:, 0].max() + zoom, 100)
    x2 = np.linspace(x[:, 1].min() - zoom, x[:, 1].max() + zoom, 100)

    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    scatter_mat(x,y)

    plt.contour(x1, x2, yp)
    plt.show()

def parte1(x,y):
    s1 = svm.SVC(kernel='linear', C=1.0)
    s1.fit(x,y)

    s2 = svm.SVC(kernel='linear', C=100.0)
    s2.fit(x,y)

    zoom = 0.4
    visualize_boundary(x,y,s1,zoom)
    visualize_boundary(x,y,s2,zoom)

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte1_2(x, y, C, sigma):
    s = svm.SVC(kernel='rbf', C=C, gamma= 1 / (2 * sigma ** 2))
    s.fit(x,y)

    visualize_boundary(x,y,s, 0.02)
    plt.show()

def parte1_3(x, xVal, y, yVal):
    cs = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigmas = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    numCs = cs.shape[0]
    numSigmas = cs.shape[0]

    res = np.zeros(numCs * numSigmas).reshape(numCs, numSigmas)

    bestDetected = 0
    bestC = -1
    bestSigma = -1
    bestSVM = {}

    for i in np.arange(numCs):
        for j in np.arange(numSigmas):
            s = svm.SVC(kernel='rbf', C=cs[i], gamma= 1 / (2 * sigmas[j] ** 2))
            s.fit(x,y)

            res[i,j] = accuracy_score(yVal, s.predict(xVal))
            if bestDetected < res[i,j]: 
                bestDetected = res[i,j]
                bestC = i
                bestSigma = j
                bestSVM = s

    print("Best accuracy: " + str(bestDetected))
    print("C: " + str(bestC))
    print("Sigma: " + str(bestSigma))
    visualize_boundary(x,y,bestSVM, 0.02)
    plt.show()

def parte2():
    return 0

def main():
    data1 = loadmat("Data/Parte1/ex6data1.mat")

    x1 = data1['X']
    y1 = data1['y']
    y1R = np.ravel(y1)

    data2 = loadmat("Data/Parte1/ex6data2.mat")

    x2 = data2['X']
    y2 = data2['y']
    y2R = np.ravel(y2)

    data3 = loadmat("Data/Parte1/ex6data3.mat")

    x3 = data3['X']
    y3 = data3['y']
    y3R = np.ravel(y3)

    x3Val = data3['Xval']
    y3Val = data3['yval']
    y3RVal = np.ravel(y3Val)

    parte1(x1, y1R)
    parte1_2(x2, y2R, 1.0, 0.1)
    parte1_3(x3, x3Val, y3, y3Val)

if __name__ == "__main__":
    main()