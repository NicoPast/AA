import numpy as np

from scipy.optimize import minimize

import time

def sigmoide(z):
    return 1 / (1 + np.exp(-z)) + 1e-9

def forwardProp(x, num_capas, thetas):
    a = np.empty(num_capas + 1, dtype="object")
    a[0] = x
    for i in range(num_capas):
        aNew = np.hstack([np.ones([x.shape[0], 1]), a[i]])
        a[i] = aNew
        a[i+1] = sigmoide(np.dot(aNew,thetas[i].T))
    return a

def coste(x, y_ones, num_capas, thetas):
    res = forwardProp(x, num_capas, thetas)[num_capas]

    return np.sum((-(y_ones) * np.log(res)) - ((1 - y_ones) * np.log(1-res))) / x.shape[0]

def costeRegul(x, y_ones, num_capas, thetas, reg):
    cost = coste(x, y_ones, num_capas, thetas)

    val = 0

    for i in range(num_capas):
        val += np.sum(np.power(thetas[i][1:], 2))

    regul = val * (reg / (2*x.shape[0]))

    return cost + regul

def grad_Delta(m, Delta, theta, reg):
    gradient = Delta

    col0 = gradient[0]
    gradient = gradient + (reg/m)*theta
    gradient[0] = col0

    return gradient

def lineal_back_prop(x, y, thetas, reg):
    m = x.shape[0]
    Delta1 = np.zeros_like(thetas[0])
    Delta2 = np.zeros_like(thetas[1])

    hThetaTot = forwardProp(x, thetas.shape[0], thetas)

    dlts = np.empty_like(thetas)
    Deltas = np.empty_like(thetas)

    dlts[-1] = (hThetaTot[-1].T - y).T
    
    for t in range(m):
        a1t = hThetaTot[0][t, :] # (401,)
        a2t = hThetaTot[1][t, :] # (26,)
        ht = hThetaTot[2][t, :] # (10,)
        yt = np.reshape(y[t], (1,)) # (10,)

        d3t = ht - yt # (10,)
        d2t = np.dot(thetas[1].T, d3t) * (a2t * (1 - a2t)) # (26,)

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    Delta1 = Delta1 / m
    Delta2 = Delta2 / m

    gradient1 = grad_Delta(m, Delta1, thetas[0], reg)
    gradient2 = grad_Delta(m, Delta2, thetas[1], reg)

    return np.append(gradient1, gradient2).reshape(-1)

def vect_back_prop(x, y, thetas, reg):
    m = x.shape[0]

    hThetaTot = forwardProp(x, thetas.shape[0], thetas)

    dlts = np.empty_like(thetas)
    Deltas = np.empty_like(thetas)

    #dlts[-1] = (hThetaTot[-1].T - y).T
    dlts[-1] = hThetaTot[-1] - y

    for i in range(1, thetas.shape[0]):
        a = hThetaTot[-(i+1)]

        delta = np.dot(thetas[-i].T, dlts[-i].T).T

        delta = delta * a * (1-a)

        delta = delta[:,1:]
        
        dlts[-(i+1)] = delta

    res = []

    for i in range(thetas.shape[0]):
        Deltas[i] = np.dot(dlts[i].T, hThetaTot[i]) / m
        Deltas[i] = np.append(Deltas[i][0], Deltas[i][1:] + (reg/m) * thetas[i][1:])
        res = np.append(res, Deltas[i]).reshape(-1)

    return res

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, x, y, reg):
    # backprop devuelve una tupla (coste, gradiente) con el coste y el gradiente de
    # una red neuronal de tres capas o mas, con num_entradas, num_ocultas nodos en las capas
    # ocultas y num_etiquetas nodos en la capa de salida. Si m es el numero de ejemplos
    # de entrenamiento, la dimension de 'X' es (m, num_entradas) y la de 'y'' es
    # (m, num_etiquetas)

    if(num_ocultas.shape[0] + 2 < 3):
        print("ERROR: num_capas incorrect, must have an input, at least one hidden and an output layer")
        return (0,0)

    # calculo de thetas
    thetas = np.empty(num_ocultas.shape[0] + 1, dtype='object')
    pointer = num_ocultas[0] * (num_entradas + 1)

    thetas[0] = np.reshape(params_rn[: pointer] , (num_ocultas[0], (num_entradas + 1)))

    for i in range(1, num_ocultas.shape[0]):
        thetas[i] = np.reshape(params_rn[pointer : pointer + num_ocultas[i] * (num_ocultas[i-1] + 1)], (num_ocultas[i], (num_ocultas[i-1] + 1)))
        pointer += num_ocultas[i] * (num_ocultas[i-1] + 1)
        
    thetas[num_ocultas.shape[0]] = np.reshape(params_rn[pointer :] , (num_etiquetas, (num_ocultas[-1] + 1)))

    #return costeRegul(x, y, thetas.shape[0], thetas, reg), lineal_back_prop(x, y, thetas, reg)
    return costeRegul(x, y, thetas.shape[0], thetas, reg), vect_back_prop(x, y, thetas, reg)

def getThetas(num_entradas, num_ocultas, num_etiquetas, out):
    thetas = np.empty(shape=[num_ocultas.shape[0] + 1], dtype='object')
    pointer = num_ocultas[0] * (num_entradas + 1)

    thetas[0] = np.reshape(out.x[: pointer] , (num_ocultas[0], (num_entradas + 1)))

    for i in range(1, num_ocultas.shape[0]):
        thetas[i] = np.reshape(out.x[pointer : pointer + num_ocultas[i] * (num_ocultas[i-1] + 1)], (num_ocultas[i], (num_ocultas[i-1] + 1)))
        pointer += num_ocultas[i] * (num_ocultas[i-1] + 1)
        
    thetas[-1] = np.array(np.reshape(out.x[pointer :] , (num_etiquetas, (num_ocultas[-1] + 1))), dtype='float')
    
    return thetas

def optm_backprop(num_entradas, num_ocultas, num_etiquetas, xTrain, xVal, yTrain, yVal, tagsTrain, tagsVal):
    print("\nCOMENCING TRAINING OF NEURONAL NETWORK\n")

    eIni = np.sqrt(6) / np.sqrt(num_etiquetas + num_entradas) # = sqrt(6) / sqrt(Lin + Lout)
    regs = np.array([0.01, 0.03, 0.1, 0.3, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    laps = np.array([25, 50, 75, 100, 250, 500, 750, 1000])

    numRegs = regs.shape[0]
    numLaps = laps.shape[0]

    resCost = np.zeros(numRegs * numLaps).reshape(numRegs, numLaps)
    resThet = np.empty_like(resCost, dtype=object)
    threads = np.empty_like(resCost, dtype=object)

    pesosSize = (num_entradas + 1) * num_ocultas[0] + (num_ocultas[-1] + 1) * num_etiquetas

    for i in range(1, num_ocultas.shape[0]):
        pesosSize = pesosSize + ((num_ocultas[i-1] + 1) * num_ocultas[i])

    startTime = time.time()

    for i in np.arange(numRegs):
        print('Testing for reg: ' + str(regs[i]))
        for j in np.arange(numLaps):
            print('Testing for laps: ' + str(laps[j]))
            pesos = np.random.uniform(-eIni, eIni, pesosSize)
            out = minimize(fun = backprop, x0= pesos, 
                args = (num_entradas, num_ocultas, num_etiquetas, xTrain, tagsTrain, regs[i]),
                method='TNC', jac = True, options = {'maxiter': laps[j]})            
            resThet[i,j] = getThetas(num_entradas, num_ocultas, num_etiquetas, out)
            resCost[i,j] = coste(xTrain, tagsTrain, resThet[i,j].shape[0], resThet[i,j])

    print(resCost)
    bestCost = np.min(resCost)
    w = np.where(resCost == bestCost)
    bestRegIndex = w[0][0]
    bestLapsIndex = w[1][0]
    bestTheta = resThet[bestRegIndex, bestLapsIndex]

    print("Best cost: " + str(bestCost))
    print("Best Reg: " + str(regs[bestRegIndex]))
    print("Best Laps: " + str(laps[bestLapsIndex]))

    endTime = time.time()
    print('Seconds elapsed of test: ' + str(endTime - startTime))

    res = forwardProp(xVal, bestTheta.shape[0], bestTheta)[-1]
    maxIndices = np.argmax(res,axis=1)
    acertados = np.sum(maxIndices == yVal)
    print("Accuracy of train: " + str(acertados*100/np.shape(res)[0]) + "%")

    return bestTheta