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

        # ajouter les labels
        # titre
        #

        plt.figure(figsize=(8, 8))
        plt.suptitle("Pair Plot")

        N = len(self.numeric_features[:3])

        for i in range(N):
            for j in range(N):
                plt.subplot(N, N, 1+j+i*N)
                if i == j:
                    h = HistogramPerHouse(path_to_data_set=self.path_to_data_set, legend=False, granularity=30)
                    h.Plot(self.numeric_features[i])
                else:
                    sc = ScatterPlotPerHouse(path_to_data_set=self.path_to_data_set, legend=False, size=1)
                    sc.Plot(self.numeric_features[i],self.numeric_features[j])





if __name__=='__main__':
    '''You have to run it with python3'''
    pp = PairPlot()
    pp.extractNumericFeatures()
    pp.Plot()
    plt.show(block=True)




