import numpy as np

import sklearn.preprocessing as sk
import scipy.optimize as opt

import time

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def costeReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return -((np.dot(np.log(h), y) + np.dot(np.log(1 - h), 1 - y)) / len(x)) + (l/(2*len(x))) * l * np.sum(thetas[1:] ** 2)

def gradienteReg(thetas, x, y, l):
    h = sigmoide(np.dot(x, thetas))
    return np.dot(x.T, h-y) / len(y) + (thetas * l) / len(y)

def evalLogisticReg(xTrain, xVal, yTrain, yVal):
    l = 1.0
    exp = 3 # 4 peta

    startTime = time.time()

    polynomial = sk.PolynomialFeatures(exp)
    xPolTrain = polynomial.fit_transform(xTrain)
    xPolVal = polynomial.fit_transform(xTrain)

    n = np.shape(xPolTrain)[1]
    thetas = np.zeros(n)

    result = opt.fmin_tnc(func=costeReg, x0=thetas, fprime=gradienteReg, args=(xPolTrain,yTrain,l))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))
    theta_opt = result[0]
    print(costeReg(theta_opt,xPolVal,yVal, l))

    print(theta_opt.shape)

    print(xPolVal.shape)

    res = sigmoide(np.dot(theta_opt, xPolVal.T))
    acertados = np.sum((res >= 0.5) == yVal)
    print("Accuracy of train: " + str(acertados*100/np.shape(res)[0]) + "%")

    return theta_opt