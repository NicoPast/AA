import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from scipy.optimize import minimize

def costeRegul(thetas, x, y, reg):
    m = np.shape(x)[0]
    return (np.sum((np.dot(x,thetas) - y.T) ** 2)) / (2 * m) + (reg / (2 * m)) * np.sum(thetas[1:] ** 2)

def gradiente(thetas, x, y):
    return (np.dot((np.dot(x, thetas) - y.T), x)) / x.shape[0]

def gradienteRegul(thetas, x, y, reg):
    grad = gradiente(thetas, x, y)
    
    res = grad + thetas * (reg / x.shape[0])
    res[0] = grad[0]
    return res

def desc_grad(thetas, x, y, reg):
    return costeRegul(thetas, x, y, reg), gradienteRegul(thetas, x, y, reg)

def parte1(x, xNew, y, n, reg):
    thetas = np.ones(n+1)

    grad = gradienteRegul(thetas, xNew, y, reg)
    print("Gradient of thetas [1,1]: " + str(grad))

    cost = costeRegul(thetas, xNew, y, reg)
    print("Cost of thetas [1,1]: " + str(cost))

    res = minimize(desc_grad, x0=thetas, args=(xNew, y, reg), jac=True, method='TNC')

    plt.figure()
    minX = np.min(x)
    maxX = np.max(x)
    minY = res.x[0] + res.x[1]*minX
    maxY = res.x[0] + res.x[1]*maxX
    plt.scatter(x, y, color='red', marker='x')
    plt.plot([minX, maxX], [minY, maxY], color='blue')
    plt.show()

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte2(x, xVal, y, yVal, reg):
    m = np.shape(x)[0]
    XvalNew = np.hstack([np.ones([np.shape(xVal)[0],1]),xVal])

    train = np.zeros(m)
    val = np.zeros(m)

    for i in np.arange(1, m+1):
        xEval = x[: i]
        yEval = y[: i]

        thetas = np.zeros(np.shape(x)[1])
        res = minimize(desc_grad, x0=thetas, args=(xEval, yEval, reg), jac=True, method='TNC')
        train[i-1] = costeRegul(res.x, xEval, yEval, reg)
        val[i-1] = costeRegul(res.x, XvalNew, yVal, reg)

    plt.figure()
    plt.title('Learning curve for linear regression')
    plt.plot(np.linspace(0, m-1, m, dtype=int), train, label='Train')
    plt.plot(np.linspace(0, m-1, m, dtype=int), val, label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')    
    plt.legend()
    plt.show()

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def createPoliX(x, p):
    xR = np.ravel(x)
    return np.array([(xR * (xR ** i)) for i in np.arange(p)]).T

def normalizaMat(mat):
    mu = np.array(np.mean(mat, axis=0))
    sigma = np.array(np.std(mat, axis=0))

    matNorm = (mat - mu) / sigma

    return matNorm, mu, sigma

def parte3(x, p, y, reg):
    matPoli = createPoliX(x, p)

    matNorm, mu, sigma = normalizaMat(matPoli)

    matNorm = np.hstack([np.ones([np.shape(matNorm)[0],1]),matNorm])

    thetas = np.zeros(matNorm.shape[1])

    res = minimize(desc_grad, x0=thetas, args=(matNorm, y, reg), jac=True, method='TNC')

    plt.plot(x, y, "x", color='red')

    margin = 5.65
    lineX = np.arange(np.min(x) - margin,np.max(x) + margin,0.05)
    valsX = (createPoliX(lineX, p)-mu) / sigma
    lineY = np.dot(np.hstack([np.ones([len(valsX),1]),valsX]), res.x)
    plt.plot(lineX, lineY, '-', c = 'blue')
    plt.show()
    
def parte3_2(x, p, xVal, y, yVal, reg):
    matNorm, mu, sigma = normalizaMat(createPoliX(x, p))
    matNorm = np.hstack([np.ones([np.shape(matNorm)[0],1]),matNorm])

    matXValNorm = (createPoliX(xVal, p) - mu) / sigma

    parte2(matNorm, matXValNorm, y, yVal, reg)

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def parte4(x, p, xVal, y, yVal):
    lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    lambdasSize = lambdas.shape[0]

    matNorm, mu, sigma = normalizaMat(createPoliX(x,p))
    matNorm = np.hstack([np.ones([np.shape(matNorm)[0],1]),matNorm])

    matValNorm = (createPoliX(xVal,p) - mu) / sigma
    matValNorm = np.hstack([np.ones([np.shape(matValNorm)[0],1]),matValNorm])

    train = np.zeros(lambdasSize)
    val = np.zeros(lambdasSize)

    for i in np.arange(lambdasSize):
        thetas = np.zeros(np.shape(matNorm)[1])
        res = minimize(desc_grad, thetas, args=(matNorm, y, lambdas[i]), jac= True, method='TNC')
        train[i] = costeRegul(res.x, matNorm, y, 0)
        val[i] = costeRegul(res.x, matValNorm, yVal, 0)

    plt.title('Selecting lambda using a cross validation set')
    plt.plot(lambdas,train,label="Train")
    plt.plot(lambdas,val,label="Cross Validation")
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')    
    plt.legend()
    plt.show()

def parte4_2(x, p, xTest, y, yTest, reg):
    matNorm, mu, sigma = normalizaMat(createPoliX(x,p))
    matNorm = np.hstack([np.ones([np.shape(matNorm)[0],1]),matNorm])

    thetas = np.ones(matNorm.shape[1])

    res = minimize(desc_grad, thetas, args=(matNorm, y, reg), jac=True, method='TNC')

    matTest = (createPoliX(xTest,p) - mu) / sigma
    matTest = np.hstack([np.ones([np.shape(matTest)[0],1]),matTest])

    error = costeRegul(res.x, matTest, yTest, 0)
    print("Error para lambda = " + str(reg) + ": " + str(error))

def main():
    data = loadmat("Data/ex5data1.mat")

    x = data['X']
    y = data['y']
    yR = np.ravel(y)

    xVal = data['Xval']
    yVal = data['yval']
    yValR = np.ravel(yVal)

    xTest = data['Xtest']
    yTest = data['ytest']

    m = np.shape(x)[0]
    n = np.shape(x)[1]

    xNew = np.hstack([np.ones([m, 1]), x])

    p = 8
    
    reg = 1
    parte1(x, xNew, yR, n, reg)
    parte2(xNew, xVal, yR, yValR, reg)

    reg = 0
    parte3(x, p, yR, reg)
    parte3_2(x, p, xVal, yR, yValR, reg)

    reg = 3
    parte4(x, p, xVal, yR, yVal)
    parte4_2(x, p, xTest, yR, yTest, reg)

if __name__ == "__main__":
    main()