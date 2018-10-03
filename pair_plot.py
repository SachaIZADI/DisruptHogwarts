import sys
import os
import matplotlib.pyplot as plt
from histogram import HistogramPerHouse
from scatter_plot import ScatterPlotPerHouse
from describe import DataSet


class PairPlot:

    def __init__(self, path_to_data_set='resources/dataset_train.csv', max_nb_features=4, fig_size=(8,8)):
        self.path_to_data_set = path_to_data_set
        self.data_set = DataSet(self.path_to_data_set)
        self.data_set.loadDataSet()
        self.max_nb_features = max_nb_features
        self.fig_size = fig_size
        self.numeric_features = []


    def extractNumericFeatures(self):
        for i in range(len(self.data_set.data_set[0])):
            if self.data_set.data_set[0][i] != 'Index' and self.data_set.isNumericFeature(i):
                self.numeric_features += [i]


    def Plot(self):

        # ajouter les labels

        plt.figure(figsize=self.fig_size)
        SMALL_SIZE = 5
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)

        plt.suptitle("Pair Plot")

        N = len(self.numeric_features[:self.max_nb_features])

        for i in range(N):
            for j in range(N):
                ax = plt.subplot(N, N, 1+j+i*N)
                if i==0:
                    ax.xaxis.set_label_position('top')
                    plt.xlabel(self.data_set.data_set[0][self.numeric_features[j]], fontsize=8)
                if j==0:
                    plt.ylabel(self.data_set.data_set[0][self.numeric_features[i]], fontsize=8)
                if i == j:
                    h = HistogramPerHouse(path_to_data_set=self.path_to_data_set, legend=False, granularity=30)
                    h.Plot(self.numeric_features[i])
                else:
                    sc = ScatterPlotPerHouse(path_to_data_set=self.path_to_data_set, legend=False, size=1)
                    sc.Plot(self.numeric_features[j],self.numeric_features[i])




if __name__=='__main__':
    '''You have to run it with python3'''

    try:
        max_nb_features = int(sys.argv[1])
        fig_size = (int(sys.argv[2]),int(sys.argv[3]))

        pp = PairPlot(max_nb_features=max_nb_features, fig_size=fig_size)
        pp.extractNumericFeatures()

    except:
        pp = PairPlot()
        pp.extractNumericFeatures()

    pp.Plot()
    plt.show(block=True)




