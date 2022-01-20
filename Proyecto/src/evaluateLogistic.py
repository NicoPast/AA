import numpy as np

import sklearn.preprocessing as sk
import scipy.optimize as opt

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import time

from threadRetVal import ThreadWithReturnValue

def sigmoide(z):
    return (1 / (1 + np.exp(-z)))

def coste(thetas, x, y):
    h = sigmoide(np.dot(x, thetas))
    return -1/len(x) * (np.dot(np.log(h), y) + np.dot(np.log(1-h), 1-y))

def costeReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return -((np.dot(np.log(h), y) + np.dot(np.log(1 - h), 1 - y)) / len(x)) + (l/(2*len(x))) * l * np.sum(thetas[1:] ** 2)

def gradienteReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(y) + (thetas * l) / len(y)

def threadMethod(xPolTrain, xPolVal, yTrain, yVal, n, exp, l):
    print('Testing for exp: ' + str(exp) + ' and lambda: ' + str(l))
    thetas = np.zeros(n)
    thetas = opt.fmin_tnc(func=costeReg, x0=thetas, disp=False, fprime=gradienteReg, args=(xPolTrain,yTrain,l))[0]
    cost = coste(thetas,xPolVal,yVal)
    print('Completed test for exp: ' + str(exp) + ' and lambda: ' + str(l))
    return thetas, cost

def evalLogisticReg(xTrain, xVal, yTrain, yVal):
    ls = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    exps = np.array([1, 2]) # 3 asks for too much memory

    numLs = ls.shape[0]
    numExps = exps.shape[0]

    pol = np.empty(numExps, dtype=object)

    for i in np.arange(numExps):
        pol[i] = sk.PolynomialFeatures(exps[i])

    startTime = time.time()

    resCost = np.zeros(numExps * numLs).reshape(numExps, numLs)
    resThet = np.empty_like(resCost, dtype=object)
    threads = np.empty_like(resCost, dtype=object) # can't be used, for lack of storage

    for i in np.arange(numExps):
        xPolTrain = pol[i].fit_transform(xTrain)
        xPolVal = pol[i].fit_transform(xVal)
        n = np.shape(xPolTrain)[1]
        for j in np.arange(numLs):
            threads[i,j] = ThreadWithReturnValue(target=threadMethod, args=(xPolTrain, xPolVal, yTrain, yVal, n, exps[i], ls[j],))
            threads[i,j].start()

    for i in np.arange(numExps):
        for j in np.arange(numLs):
            resThet[i,j], resCost[i,j] = threads[i,j].join()

    bestCost = np.min(resCost)
    w = np.where(resCost == bestCost)
    print(bestCost)
    print(resCost)
    bestExpIndex = w[0][0]
    bestLIndex = w[1][0]
    bestTheta = resThet[bestExpIndex, bestLIndex]

    print()
    print('Best cost: ' + str(bestCost))
    print('Best lambda: ' + str(ls[bestLIndex]))
    print('Best exponent: ' + str(exps[bestExpIndex]))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))
    print()

    fig = plt.figure()

    expexp, ll = np.meshgrid(ls, exps)

    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_xlabel('Exponent')
    ax.set_ylabel('L')
    ax.set_zlabel('Cost')

    fig.add_axes(ax)

    ax.plot_surface(ll,expexp,resCost, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.savefig('../Results/LogReg/LOG_1.png', bbox_inches='tight')
    plt.close()

    plt.scatter(expexp, ll, s=80, c=resCost)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Exponent')
    plt.clim(np.min(resCost), np.max(resCost))
    plt.colorbar().set_label('Cost')
    plt.scatter(ls[bestLIndex], exps[bestExpIndex], s=40, marker='x', color='r')
    plt.savefig('../Results/LogReg/LOG_2.png', bbox_inches='tight')
    plt.close()

    xPolVal = pol[bestExpIndex].fit_transform(xVal)

    res = sigmoide(np.dot(bestTheta, xPolVal.T))
    acertados = np.sum((res >= 0.5) == yVal)
    accuracy = acertados*100/np.shape(res)[0]
    print("Accuracy of train: " + str(accuracy) + "%")

    return bestTheta, pol[bestExpIndex], accuracy, bestCost, ls[bestLIndex], exps[bestExpIndex]

def getNumAcertadosLog(theta, x, y):
    res = sigmoide(np.dot(theta, x.T))
    return np.sum((res >= 0.5) == y)