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
    ax = Axes3D(fig)

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

def minimizeCost(x, y):

    m = len(x)
    alpha = 0.01

    theta0 = theta1 = 0

    # print (coste(theta0, theta1, x, y, m))
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

 

    print(theta0)
    print(theta1)

    print(theta0 + theta1*7)

    return [theta0, theta1]

def coste (theta0, theta1, x, y, m):
    return np.sum(((theta0 + theta1 * x) - y) ** 2) / (2 * m)

def descensoGradiente(x, y, alpha):
    normalizaMat(x)

def normalizaMat(mat):
    mu = np.array(np.mean(mat, axis=0))
    sigma = np.array(np.std(mat, axis=0))
    
    matNorm = ones_like(mat)

    print(mu)
    print(sigma)

    #print(mat)

    for i in np.arange(np.shape(mat)[1]-1):
        #print(i+1)
        matNorm[:, i+1] = (mat[:, i+1] - mu[i+1])/ sigma[i+1]
    #matNorm = (mat - mu) / sigma

    print(matNorm)

    return matNorm, mu, sigma


def parte1():
    data = loadCSV("ex1data1.csv")

    x = data[:, 0]
    y = data[:, 1]

    t0, t1 = minimizeCost(x, y)
    
    t0Mat, t1Mat, costeMat = make_data([-10,10], [-1,4],x, y)
    paint(x, y, t0, t1, t0Mat, t1Mat, costeMat)

def parte2():
    data = loadCSV("ex1data2.csv")

    x = data[:, :-1]
    y = data[:, -1]
    
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    
    x = np.hstack([np.ones([m, 1]), x])
    x, mu, sigma = normalizaMat(x)
    # mu = np.hstack([1, mu])
    # sigma = np.hstack([0, sigma])
    
    alpha = 0.01

    #descensoGradiente(x, y, alpha)


if __name__ == "__main__":
    # parte1()
    parte2()



