import numpy as np
import matplotlib.pyplot as plt

from pandas.io.parsers import read_csv

from prepareData import analyzeData, prepareData

def loadCSV(fileName):
    data = read_csv(fileName, sep=';',  on_bad_lines='skip')
    data.fillna('empty', inplace=True)
    prepareData(data)
    return data

def main():
    dataR = loadCSV("Data/MushroomDataset/secondary_data.csv")
    
    dataR.fillna('empty', inplace=True)

    analyzeData(dataR)

    data = dataR.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]

if __name__ == "__main__":
    main()