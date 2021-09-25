import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def loadCSV(fileName):
    return read_csv(fileName, header=None).to_numpy().astype(float)

def minimizeCost(fileName):
    data = loadCSV(fileName)

    x = data[:, 0]
    y = data[:, 1]

    m = len(x)
    alpha = 0.01

    theta0 = theta1 = 0
    for _ in range(1500):
        """
            Esto visualiza la progresion de la pendiente gradualmente en funcion de los erroes
        """
        minX = min(x)
        maxX = max(x)
        minY = theta0 + theta1 * minX
        maxY = theta0 + theta1 * maxX
        plt.plot([minX, maxX], [minY, maxY], "--", linewidth=0.5)

        """
            Ajusta una vez la theta0 y theta1 segun el error calculado
        """
        sum0 = np.sum((x * theta1 + theta0) - y)
        sum1 = np.sum(((x * theta1 + theta0) - y) * x)

        theta0 = theta0 - (alpha/m) * sum0
        theta1 = theta1 - (alpha/m) * sum1

    
    minX = min(x)
    maxX = max(x)
    minY = theta0 + theta1 * minX
    maxY = theta0 + theta1 * maxX
    plt.plot([minX, maxX], [minY, maxY])
    plt.plot(x, y, "x")
    plt.show()





if __name__ == "__main__":
    minimizeCost("ex1data1.csv")