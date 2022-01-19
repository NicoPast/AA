import numpy as np

import sklearn.svm as svm

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import accuracy_score

import time

from threadRetVal import ThreadWithReturnValue

def threadMethod(x, xVal, y, yVal, c, sigma):
    print('Testing c ' + str(c) + ' sigma ' + str(sigma))
    s = svm.SVC(kernel='rbf', C=c, gamma= 1 / (2 * sigma ** 2))
    s.fit(x,y)

    print('Completed c ' + str(c) + ' sigma ' + str(sigma))
    return s, accuracy_score(yVal, s.predict(xVal))

def evaluateSVM(x, xVal, y, yVal):
    print("\nCOMENCING TRAINING OF SVM\n")

    startTime = time.time()

    cs = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigmas = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    numCs = cs.shape[0]
    numSigmas = cs.shape[0]

    resAcc = np.zeros(numCs * numSigmas).reshape(numCs, numSigmas)
    resSVM = np.empty_like(resAcc, dtype=object)
    threads = np.empty_like(resAcc, dtype=object)

    bestAcc = 0
    bestC = -1
    bestSigma = -1
    bestSVM = {}

    for i in np.arange(numCs):
        for j in np.arange(numSigmas):
            threads[i,j] = ThreadWithReturnValue(target=threadMethod, args=(x, xVal, y, yVal, cs[i], sigmas[j],))
            threads[i,j].start()

    for i in np.arange(numCs):
        for j in np.arange(numSigmas):
            resSVM[i,j], resAcc[i,j] = threads[i,j].join()
    
    bestAcc = np.max(resAcc)
    w = np.where(resAcc == bestAcc)
    print(w)
    bestC = cs[w[0][0]]
    bestSigma = sigmas[w[1][0]]
    bestSVM = resSVM[w[0][0],w[1][0]]

    print("Best accuracy: " + str(bestAcc))
    print("Best C: " + str(bestC))
    print("Best Sigma: " + str(bestSigma))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))
    
    fig = plt.figure()

    cc, ss = np.meshgrid(sigmas, cs)

    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_xlabel('C')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Accuracy')

    fig.add_axes(ax)

    ax.plot_surface(ss,cc,resAcc, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()

    plt.scatter(ss, cc, c=resAcc)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('C')
    plt.ylabel('Sigma')
    plt.clim(np.min(resAcc), np.max(resAcc))
    plt.colorbar().set_label('Accuracy')
    plt.scatter(bestC, bestSigma, marker='x', color='r')
    plt.show()

    return bestSVM, bestAcc, bestC, bestSigma