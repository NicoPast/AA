from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from pandas.io.parsers import read_csv

def loadCSVRaw(fileName):
    return read_csv(fileName, sep=';', on_bad_lines='skip')

def loadCSV(fileName):
    return loadCSVRaw(fileName).to_numpy()

def analyzeData(fileName):
    data = loadCSVRaw(fileName)

    data['class'] = [int(i == 'p') for i  in data['class']]

    class CapShape(Enum):
        b = 0 
        c = 1
        x = 2
        f = 3
        s = 4
        p = 5
        o = 6

    data['cap-shape'] = [CapShape[i].value for i in data['cap-shape']]

    class CapSurface(Enum):
        nan = -1
        i = 0
        g = 1
        y = 2
        s = 3
        h = 4
        l = 5
        k = 6
        t = 7
        w = 8
        e = 9

    print(data['cap-surface'])

    data['cap-surface'] = [CapSurface[i].value for i in data['cap-surface']]

    print(data.head())

    print(data.describe())

    # data.hist(figsize=(10, 10))
    # plt.tight_layout()

    print(data.corr()['class'])

    plt.figure(figsize=(6, 8))
    sns.heatmap(data.corr()[['class']], annot=True, vmin=-1, vmax=1)
    plt.show()

def main():
    analyzeData("Data/MushroomDataset/secondary_data.csv")

    data1 = loadCSV("Data/MushroomDataset/primary_data.csv")
    data2 = loadCSV("Data/MushroomDataset/secondary_data.csv")

    print(data1.shape)
    print(data2.shape)

if __name__ == "__main__":
    main()