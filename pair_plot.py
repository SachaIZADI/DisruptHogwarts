import sys
import matplotlib.pyplot as plt
from histogram import HistogramPerHouse
from scatter_plot import ScatterPlotPerHouse
from describe import DataSet

# TODO : save l'image totale dans un fichier


class PairPlot:
    """
    - A class to plot the pair plot of many Hogwarts features.
    - Example to run:
        from pair_plot import PairPlot
        import matplotlib.pyplot as plt
        pp = PairPlot()
        pp.Plot()
        plt.show()
    """

    def __init__(self, path_to_data_set='resources/dataset_train.csv', max_nb_features=4, fig_size=(8,8)):
        """
        :param path_to_data_set: a string. The path to the dataset.
        :param max_nb_features: an integer. The number of features to analyze - analysis will start from the first feature (on the left) and continue until reaching the number max of features.
                                This was necessary for the sake of readability (there are ~10 numeric features, which would lead to 10**2 = 100 plots to do.
        :param fig_size: an integer tuple. The size of the figure to output.
        """
        self.path_to_data_set = path_to_data_set
        self.data_set = DataSet(self.path_to_data_set)
        self.data_set.loadDataSet()
        self.max_nb_features = max_nb_features
        self.fig_size = fig_size
        self.numeric_features = []


    def extractNumericFeatures(self):
        """
        Automatically extracts the numeric features in the dataset.
        """
        for i in range(len(self.data_set.data_set[0])):
            if self.data_set.data_set[0][i] != 'Index' and self.data_set.isNumericFeature(i):
                self.numeric_features += [i]


    def Plot(self):
        """
        Plotting function.
        :return:
        """
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

        handles, labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower right', prop={'size': 6})


if __name__=='__main__':
    '''
    - How to run it: 
        python3 pair_plot.py
        python3 pair_plot.py 4 9 9
    - /!\ Make sure to use python3 and not python2 /!\ 
    '''
    try:
        max_nb_features = int(sys.argv[1])
        fig_size = (int(sys.argv[2]), int(sys.argv[3]))

        pp = PairPlot(max_nb_features=max_nb_features, fig_size=fig_size)
        pp.extractNumericFeatures()

    except:

        pp = PairPlot(max_nb_features=13,fig_size=(30, 30))
        pp.extractNumericFeatures()
        pp.Plot()
        plt.savefig('img/full_pair_plot.png')
        plt.clf()

        pp = PairPlot()
        pp.extractNumericFeatures()

    pp.Plot()
    plt.show(block=True)




