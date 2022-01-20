import numpy as np
import matplotlib.pyplot as plt

import sklearn.svm as svm
from sklearn.metrics import accuracy_score

from pandas.io.parsers import read_csv

from prepareData import analyzeData, prepareData
from evaluateSVM import evaluateSVM, evaluateSVMLinear
from evaluateNeuronal import evalNN, getNumAcertadosNN
from evaluateLogistic import evalLogisticReg, getNumAcertadosLog

from threadRetVal import ThreadWithReturnValue

import time

def loadCSV(fileName):
    data = read_csv(fileName, sep=';', on_bad_lines='skip')
    data.fillna('empty', inplace=True)
    prepareData(data)
    return data

def evalLogistic(xTrain, xVal, xTest, yTrain, yVal, yTest):
    print("\nCOMENCING TRAINING OF LOGISTIC REGRESION\n")

    th, pol, acc, cost, l, exp = evalLogisticReg(xTrain, xVal, yTrain, yVal)

    xPolTest = pol.fit_transform(xTest)

    acertados = getNumAcertadosLog(th, xPolTest, yTest)
    accuracy = acertados*100/np.shape(xTest)[0]
    print('Accuracy over Test sample: ' + str(accuracy) + "%")

    return accuracy, acc, cost, l, exp

def evalNueronalThread(ocultas, num_entradas, num_etiquetas, xTrain, xVal, xTest, yTrain, yVal, yTest, tagsTrain, tagsVal):
    print("\nTesting with " + str(ocultas))
    num_ocultas = np.array(ocultas)
    th, acc, cost, reg, laps = evalNN( 
    num_entradas, num_ocultas, num_etiquetas, 
    xTrain, xVal, yTrain, yVal, tagsTrain, tagsVal)

    acertados = getNumAcertadosNN(xTest, yTest, th)

    accuracy = acertados*100/np.shape(xTest)[0]
    print("Accuracy over Test sample: " + str(accuracy) + "%")
    return accuracy, acc, cost, reg, laps

def evalNeuronal(xTrain, xVal, xTest, yTrain, yVal, yTest, n):
    print("\nCOMENCING TRAINING OF NEURONAL NETWORK\n")

    ocultas = np.array([[20], [40], [60], [80], [100], [120], [60, 20], [60, 40]])
    numOcultas = ocultas.shape[0]

    resAcc = np.zeros(numOcultas)
    resAccVal = np.zeros(numOcultas)
    resCost = np.zeros(numOcultas)
    resReg = np.zeros(numOcultas)
    resLaps = np.zeros(numOcultas)

    startTime = time.time()

    tagsTrain = np.zeros((len(yTrain), 2))
    for i in range(len(yTrain)):
        tagsTrain[i][int(yTrain[i])] = 1

    tagsVal = np.zeros((len(yVal), 2))
    for i in range(len(yVal)):
        tagsVal[i][int(yVal[i])] = 1

    num_etiquetas = 2
    num_entradas = n

    threads = np.empty(numOcultas, dtype=object)

    for i in np.arange(numOcultas):
        threads[i] = ThreadWithReturnValue(target=evalNueronalThread, args=(ocultas[i], num_entradas, num_etiquetas, xTrain, xVal, xTest, yTrain, yVal, yTest, tagsTrain, tagsVal,))
        threads[i].start()

    for i in np.arange(numOcultas):
        resAcc[i], resAccVal[i], resCost[i], resReg[i], resLaps[i] = threads[i].join()

    bestAcc = np.max(resAcc)
    bestOcIndex = np.where(bestAcc == resAcc)[0]
    bestCost = resCost[bestOcIndex]
    bestReg = resReg[bestOcIndex]
    bestLaps = resLaps[bestOcIndex]
    bestAccVal = resAccVal[bestOcIndex]

    print()
    print('Best neurons in hidden layer: ' + str(ocultas[bestOcIndex]))
    print('Best Accuracy: ' + str(bestAcc) + '%')
    print("Accuracy of train: " + str(bestAccVal) + "%")
    print('Best cost: ' + str(bestCost))
    print('Best reg: ' + str(bestReg))
    print('Best laps: ' + str(bestLaps))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))

    ocultasStr = []
    for i in np.arange(numOcultas):
        ocultasStr.append(str(ocultas[i]))

    plt.plot(np.arange(numOcultas), resAcc)
    plt.xticks(np.arange(numOcultas), ocultasStr)
    plt.savefig('../Results/NeuronalNetwork/NN.png', bbox_inches='tight')
    plt.close()

    return ocultas[bestOcIndex], bestAcc, bestAccVal, bestCost, bestReg, bestLaps

def evalSVM(xTrain, xVal, xTest, yTrain, yVal, yTest):
    print("\nCOMENCING TRAINING OF SVM\n")

    s, acc, c, sig = evaluateSVM(xTrain, xVal, yTrain, yVal)
    accuracy = accuracy_score(yTest, s.predict(xTest)) * 100
    print('Accuracy over Test sample: ' + str(accuracy) + '%')

    return accuracy, acc, c, sig

def evalSVMLinear(xTrain, xVal, xTest, yTrain, yVal, yTest):
    print("\nCOMENCING TRAINING OF SVM LINEAR\n")

    s, acc, c = evaluateSVMLinear(xTrain, xVal, yTrain, yVal)
    accuracy = accuracy_score(yTest, s.predict(xTest)) * 100
    print('Accuracy over Test sample: ' + str(accuracy) + '%')
    return accuracy, acc, c

def createDataSets(x, y, m, trainPerc, valPerc, testPerc):
    if(trainPerc + valPerc + testPerc > 1.0):
        print("ERROR: Percentages given not valid")
        exit(-1)

    print('Size of Training set: ' + str(int(trainPerc * m)))
    print('Size of Validation set: ' + str(int(valPerc * m)))
    print('Size of Teseting set: ' + str(int(testPerc * m)))
    
    valPerc += trainPerc
    testPerc += valPerc

    train = int(trainPerc * m)
    val = int(valPerc * m)
    test = int(testPerc * m)

    xTrain = x[:train]
    yTrain = y[:train]

    xVal = x[train:val]
    yVal = y[train:val]

    xTest = x[val:test]
    yTest = y[val:test]

    return xTrain, yTrain, xVal, yVal, xTest, yTest

def main():
    dataR = loadCSV("../Data/MushroomDataset/secondary_data_shuffled.csv")

    #analyzeData(dataR)

    data = dataR.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]

    m = y.shape[0]
    n = x.shape[1]

    trainPerc = 0.2
    valPerc = 0.2
    testPerc = 0.2

    # 0.2 0.2 0.2
    # Seconds taken for the evaluation: 8767.091146945953
    # Minutes taken for the evaluation: 146.11818578243256
    # Hours taken for the evaluation: 2.4353030963738758

    xTrain, yTrain, xVal, yVal, xTest, yTest = createDataSets(x, y, m, trainPerc, valPerc, testPerc)

    startTime = time.time()

    logAcc, logAccVal, logCost, logL, logExp = evalLogistic(xTrain, xVal, xTest, yTrain, yVal, yTest)

    nnHidenLayer, nnAcc, nnAccVal, nnCost, nnReg, nnLaps = evalNeuronal(xTrain, xVal, xTest, yTrain, yVal, yTest, n)
    
    svmAcc, svmAccVal, svmC, svmSig = evalSVM(xTrain, xVal, xTest, yTrain, yVal, yTest)

    print('\nRESULTS OF LOGISTIC REGRESSION\n')
    print("Best accuracy: " + str(logAcc))
    print('Best cost: ' + str(logCost))
    print('Best lambda: ' + str(logL))
    print('Best exponent: ' + str(logExp))

    print('\nRESULTS OF NEURONAL NETWORK\n')
    print('Best Accuracy: ' + str(nnAcc) + '%')
    print("Accuracy of train: " + str(nnAccVal) + "%")
    print('Best neurons in hidden layer: ' + str(nnHidenLayer))
    print('Best cost: ' + str(nnCost))
    print('Best reg: ' + str(nnReg))
    print('Best laps: ' + str(nnLaps))

    print('\nRESULTS OF SVM\n')
    print("Best accuracy: " + str(svmAcc))
    print("Accuracy of train: " + str(svmAccVal) + "%")
    print("Best C: " + str(svmC))
    print("Best Sigma: " + str(svmSig))

    endTime = time.time()
    deltaTime = endTime - startTime

    print()
    print('Seconds taken for the evaluation: ' + str(deltaTime))
    print('Minutes taken for the evaluation: ' + str(deltaTime/60))
    print('Hours taken for the evaluation: ' + str(deltaTime/3600))

    plt.figure(figsize=(12, 10))
    plt.bar(np.arange(3), [logAcc, nnAcc, svmAcc])
    plt.title('Accuracy comparasion between learning algorithms')
    plt.ylabel('Accuracy')
    xlabels = ['Logistic Regresion\nAccuracy: ' + str(logAcc) + '%', 'Neural Network\nAccuracy :' + str(nnAcc) + '%', 'SVM\nAccuracy: ' + str(svmAcc) + '%']
    plt.xticks(np.arange(3), xlabels)
    plt.savefig('../Results/FinalGraph.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()