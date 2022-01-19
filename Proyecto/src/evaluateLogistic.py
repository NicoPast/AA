import numpy as np

import sklearn.preprocessing as sk
import scipy.optimize as opt

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import time

def sigmoide(z):
    return (1 / (1 + np.exp(-z))) + 1e-9

def coste(thetas, x, y):
    h = sigmoide(np.dot(x, thetas))
    return -1/len(x) * (np.dot(np.log(h), y) + np.dot(np.log(1-h), 1-y))

def costeReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return -((np.dot(np.log(h), y) + np.dot(np.log(1 - h), 1 - y)) / len(x)) + (l/(2*len(x))) * l * np.sum(thetas[1:] ** 2)

def gradienteReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(y) + (thetas * l) / len(y)

def evalLogisticReg(xTrain, xVal, yTrain, yVal):
    ls = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    exps = np.array([1, 2, 3]) # 4 peta

    numLs = ls.shape[0]
    numExps = exps.shape[0]

    pol = np.empty(numExps, dtype=object)

    for i in np.arange(numExps):
        pol[i] = sk.PolynomialFeatures(exps[i])

    startTime = time.time()

    resCost = np.zeros(numExps * numLs).reshape(numExps, numLs)
    resThet = np.empty_like(resCost, dtype=object)

    print(resCost.shape)

    for i in np.arange(numExps):
        print('Testing for exp: ' + str(exps[i]))
        xPolTrain = pol[i].fit_transform(xTrain)
        xPolVal = pol[i].fit_transform(xVal)
        n = np.shape(xPolTrain)[1]
        for j in np.arange(numLs):
            print('Testing for l: ' + str(ls[j]))
            thetas = np.zeros(n)
            resThet[i,j] = opt.fmin_tnc(func=costeReg, x0=thetas, disp=False, fprime=gradienteReg, args=(xPolTrain,yTrain,ls[j]))[0]
            resCost[i,j] = coste(resThet[i,j],xPolVal,yVal)

    bestCost = np.min(resCost)
    w = np.where(resCost == bestCost)
    bestExpIndex = w[0][0]
    bestLIndex = w[1][0]
    bestTheta = resThet[bestExpIndex, bestLIndex]

    print('Best cost: ' + str(bestCost))
    print('Best lambda: ' + str(ls[bestLIndex]))
    print('Best exponent: ' + str(exps[bestExpIndex]))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))

    fig = plt.figure()

    expexp, ll = np.meshgrid(ls, exps)

    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_xlabel('L')
    ax.set_ylabel('Exponent')
    ax.set_zlabel('Cost')

    fig.add_axes(ax)

    ax.plot_surface(ll,expexp,resCost, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()

    plt.scatter(expexp, ll, s=80, c=resCost)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Exponent')
    plt.clim(np.min(resCost), np.max(resCost))
    plt.colorbar().set_label('Cost')
    plt.scatter(ls[bestLIndex], exps[bestExpIndex], s=40, marker='x', color='r')
    plt.show()

    xPolVal = pol[bestExpIndex].fit_transform(xVal)

    res = sigmoide(np.dot(bestTheta, xPolVal.T))
    acertados = np.sum((res >= 0.5) == yVal)
    print("Accuracy of train: " + str(acertados*100/np.shape(res)[0]) + "%")

    return bestTheta, pol[bestExpIndex]