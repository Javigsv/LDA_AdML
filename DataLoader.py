import pandas as pd
import numpy as np

class DataLoader:
    '''
    To load the data
    create a data Loader and then call load()

    dl = DataLoader(filename)
    data, V = dl.load()

    '''

    def __init__(self, filename):
        self.filename = filename

    def load(self):

        data = []

        file = open(self.filename, 'r')

        rawData = file.readlines()

        rawData = list(filter(lambda a: a != '\n', rawData))

        for rawLine in rawData:

            newLine = self.preprocess(rawLine)

            data.append(newLine)
        
        V = self.count_vocabulary(data)

        return data, V

    def preprocess(self, rawLine):

        newLine = rawLine

        newLine = newLine.replace('\n','')

        newLine = newLine.split(',')

        newLine = list(map(int, newLine))

        return newLine

    def count_vocabulary(self, data):
        max = -1
        for doc in data:
            for ind in doc:
                if ind > max:
                    max = ind

        return max

filename = './Code/Reuters_Corpus_Vectorized.csv'
dl = DataLoader(filename)
dl.load()