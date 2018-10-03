import sys
import os
import matplotlib.pyplot as plt
from histogram import HistogramPerHouse
from scatter_plot import ScatterPlotPerHouse
from describe import DataSet


class PairPlot:

    def __init__(self, path_to_data_set='resources/dataset_train.csv'):
        self.path_to_data_set = path_to_data_set
        self.data_set = DataSet(self.path_to_data_set)
        self.data_set.loadDataSet()
        self.numeric_features = []


    def extractNumericFeatures(self):
        for i in range(len(self.data_set.data_set[0])):
            if self.data_set.data_set[0][i] != 'Index' and self.data_set.isNumericFeature(i):
                self.numeric_features += [i]


    def Plot(self):


        # ca merde ici

        plt.figure(figsize=(20, 20))

        N = len(self.numeric_features[:3])

        for i in self.numeric_features[:3]:
            for j in self.numeric_features:
                plt.subplot(N, N, j+i*N)
                if i == j:
                    h = HistogramPerHouse(path_to_data_set=self.path_to_data_set, legend=False)
                    h.Plot(i)
                else:
                    sc = ScatterPlotPerHouse(path_to_data_set=self.path_to_data_set, legend=False)
                    sc.Plot(i,j)


        plt.show(block=True)



if __name__=='__main__':
    '''You have to run it with python3'''
    pp = PairPlot()
    pp.extractNumericFeatures()
    pp.Plot()
    plt.show(block=True)




