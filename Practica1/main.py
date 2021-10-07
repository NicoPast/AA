import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones_like, zeros_like
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter 

def loadCSV(fileName):
    return read_csv(fileName, header=None).to_numpy().astype(float)

def make_data (t0_range, t1_range, x, y):
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(Theta0[ix, iy], Theta1[ix, iy], x, y, len(x))

    return [Theta0, Theta1, Coste]
    
def paint(x, y, t0, t1, t0Mat, t1Mat, costeMat):
    fig = plt.figure()

    # to avoid a warning we do...
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    surf = ax.plot_surface(t0Mat,t1Mat,costeMat, cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()
    
    plt.scatter(t0, t1, marker='x', color='red')
    plt.contour(t0Mat,t1Mat,costeMat, np.logspace(-2,3,20), colors='blue')
    plt.show()
   
    minX = min(x)
    maxX = max(x)
    minY = t0 + t1 * minX
    maxY = t0 + t1 * maxX
    plt.plot([minX, maxX], [minY, maxY])
    plt.plot(x, y, "x", color='red')
    plt.show()

    print([t0, t1])

def minimizeCost(x, y):

    m = len(x)
    alpha = 0.01

    theta0 = theta1 = 0

    for _ in range(1500):
        """
            Esto visualiza la progresion de la pendiente gradualmente en funcion de los erroes
        """
        # minX = min(x)
        # maxX = max(x)
        # minY = theta0 + theta1 * minX
        # maxY = theta0 + theta1 * maxX
        # plt.plot([minX, maxX], [minY, maxY], "--", linewidth=0.5)

        """
            Ajusta una vez la theta0 y theta1 segun el error calculado
        """
        sum0 = np.sum((x * theta1 + theta0) - y)
        sum1 = np.sum(((x * theta1 + theta0) - y) * x)

        theta0 = theta0 - (alpha/m) * sum0
        theta1 = theta1 - (alpha/m) * sum1 

    # 70.000 habs = 4.53...
    # print(theta0 + theta1*7)

    return [theta0, theta1]

def coste (theta0, theta1, x, y, m):
    return np.sum(((theta0 + theta1 * x) - y) ** 2) / (2 * m)

def parte1():
    data = loadCSV("Data/ex1data1.csv")

    x = data[:, 0]
    y = data[:, 1]

    t0, t1 = minimizeCost(x, y)
    
    t0Mat, t1Mat, costeMat = make_data([-10,10], [-1,4],x, y)
    paint(x, y, t0, t1, t0Mat, t1Mat, costeMat)

"""
    ===============================================
    ===============================================
    ===============================================
    ===============================================
"""

def normalizaMat(mat):
    mu = np.array(np.mean(mat, axis=0))
    sigma = np.array(np.std(mat, axis=0))
    
    # res = (mat - mu) / sigma
    # np.nan_to_num(res, False, 1.0)

    # [1, 1541.3, 4]
    matNorm = ones_like(mat)
    # [0, 1]
    for i in np.arange(np.shape(mat)[1]-1):
        # [1, 2]
        matNorm[:, i+1] = (mat[:, i+1] - mu[i+1]) / sigma[i+1]

    return matNorm, mu, sigma

def costeVec(x, y, thetas):
    xTh = np.dot(x, thetas)
    return np.sum((xTh - y)**2) / (2*len(x))

def descensoGradiente(x, y, alpha):
    m = np.shape(x)[0]
    n = np.shape(x)[1]

    #thetas = np.zeros(n)
    thetas2 = np.zeros(n)

    costes = np.zeros(1500)

    for i in range(len(costes)):
        xTh = np.dot(x, thetas2)

        NuevaTheta = thetas2
        Aux = xTh - y
        for j in range(n):
            Aux_j = Aux * x[:, j]
            NuevaTheta[j] -= (alpha / m) * Aux_j.sum()
        costes[i] = costeVec(x, y, thetas2)

        thetas2 = NuevaTheta

    # for i in range(len(costes)):
    #     xTh = np.dot(x, thetas)

    #     temp = np.dot(np.transpose(x), (xTh - y))
    #     thetas = thetas - (alpha/m) * temp
    #     costes[i] = costeVec(x, y, thetas)

    plt.plot(np.arange(len(costes)), costes)

    plt.show()

    return thetas2
    
def ecuacionNormal(x,y):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), y))

def parte2():
    data = loadCSV("Data/ex1data2.csv")

    x = data[:, :-1]
    y = data[:, -1]
    
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    
    xNew = np.hstack([np.ones([m, 1]), x])

    xNorm, mu, sigma = normalizaMat(xNew)

    alpha = 0.01
    thetasDG = descensoGradiente(xNorm, y, alpha)
    thetasEN = ecuacionNormal(xNew,y)

    example = [1.0, 1650.0, 3.0]
    exampleNorm = ones_like(example)

    # for i in np.arange(len(example)-1):
    #     exampleNorm[i+1] = (example[i+1] - mu[i+1]) / sigma[i+1]

    exampleNorm[1:] = (example[1:] - mu[1:]) / sigma[1:]


if __name__ == "__main__":
    parte1()
    parte2()