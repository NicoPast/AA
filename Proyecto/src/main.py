import numpy as np
import matplotlib.pyplot as plt

import sklearn.svm as svm
from sklearn.metrics import accuracy_score

from pandas.io.parsers import read_csv

from prepareData import analyzeData, prepareData
from evaluateSVM import evaluateSVM
from evaluateNeuronal import optm_backprop, forwardProp
from evaluateLogistic import evalLogisticReg

import time

def loadCSV(fileName):
    data = read_csv(fileName, sep=';',  on_bad_lines='skip')
    data.fillna('empty', inplace=True)
    prepareData(data)
    return data

def evalSVM(xTrain, xVal, xTest, yTrain, yVal, yTest):
    s, acc, c, sig = evaluateSVM(xTrain, xVal, yTrain, yVal)
    print('Accuracy over Test sample: ' + str(accuracy_score(yTest, s.predict(xTest)) * 100) + '%')

def evalNeuronal(xTrain, xVal, xTest, yTrain, yVal, yTest, n):
    num_etiquetas = 1
    num_ocultas = np.array([60, 20])
    num_entradas = n

    eIni = 0.12
    laps = 70
    reg = 1

    th = optm_backprop(eIni, 
    num_entradas, num_ocultas, num_etiquetas, 
    xTrain, xVal, yTrain, yVal, 
    laps, reg)

    res = forwardProp(xTest, th.shape[0], th)[-1]
    maxIndices = np.argmax(res,axis=1) + 1 
    acertados = np.sum(maxIndices == yTest)
    print("Accuracy over Test sample: " + str(acertados*100/np.shape(res)[0]) + "%")

def evalLogistic(xTrain, xVal, xTest, yTrain, yVal, yTest):
    evalLogisticReg(xTrain, xVal, yTrain, yVal)

def main():
    dataR = loadCSV("../Data/MushroomDataset/secondary_data_shuffled.csv")
    
    dataR.fillna('empty', inplace=True)

    #analyzeData(dataR)

    data = dataR.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]

    m = y.shape[0]
    n = x.shape[1]

    trainPerc = 0.4
    valPerc = 0.2 + trainPerc
    testPerc = 0.2 + valPerc
    train = int(trainPerc * m)
    val = int(valPerc * m)
    test = int(testPerc * m)

    xTrain = x[:train]
    yTrain = y[:train]

    xVal = x[train:val]
    yVal = y[train:val]

    xTest = x[val:test]
    yTest = y[val:test]

    
    print(xTrain.shape)
    print(yTrain.shape)

    print(xVal.shape)
    print(yVal.shape)

    #evalSVM(xTrain, xVal, xTest, yTrain, yVal, yTest)

    # 0.02 0.02 0.02
    # Best accuracy: 0.7493857493857494
    # C: 10.0
    # Sigma: 10.0
    # Accuracy over Test sample: 13.338788870703763%

    # 0.1 0.1 0.1
    # Best accuracy: 0.5716391026690683
    # C: 0.1
    # Sigma: 30.0

    # 0.2 0.2 0.2
    # Best accuracy: 0.6571966595709842
    # C: 0.1
    # Sigma: 10.0

    # 0.2 0.2 0.2 Shuffled
    # Best accuracy: 0.9997543802194203
    # C: 3.0
    # Sigma: 3.0
    # Seconds elapsed of test: 5572.425535917282
    # Accuracy over Test sample: 99.98362534796136%

    # 0.2 0.2 0.2 Shuffled Thread
    # Best accuracy: 0.9997543802194203
    # C: 3.0
    # Sigma: 3.0
    # Seconds elapsed of test: 3856.3216075897217
    # Accuracy over Test sample: 99.98362534796136%

    evalNeuronal(xTrain, xVal, xTest, yTrain, yVal, yTest, n)

    #evalLogistic(xTrain, xVal, xTest, yTrain, yVal, yTest)

if __name__ == "__main__":
    main()