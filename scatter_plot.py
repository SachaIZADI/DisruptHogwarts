import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from describe import DataSet, Statistics


class ScatterPlotPerHouse:

    def __init__(self, path_to_data_set='resources/dataset_train.csv'):
        self.data_set = DataSet(path_to_data_set)
        self.data_set.loadDataSet()

    def Plot(self, col_nb_1, col_nb_2):
        feature_1 = self.data_set.extractColumn(col_nb=col_nb_1, convert_to_float=True)[1:]
        feature_2 = self.data_set.extractColumn(col_nb=col_nb_2, convert_to_float=True)[1:]
        houses = self.data_set.extractColumn(col_nb=1)[1:]

        to_plot = {
                    'feature_1':{},
                    'feature_2':{},
                   }

        for i in range(len(houses)):
            try:
                to_plot['feature_1'][houses[i]] += [feature_1[i]]
            except:
                to_plot['feature_1'][houses[i]] = [feature_1[i]]

            try:
                to_plot['feature_2'][houses[i]] += [feature_2[i]]
            except:
                to_plot['feature_2'][houses[i]] = [feature_2[i]]

        unique_houses = set(houses)

        """

        full_list = []
        
        for house in unique_houses:
            full_list += to_plot[house]

        s = Statistics(full_list)
        min = s.Quartile(0)
        max = s.Quartile(1)
        bins = np.linspace(min, max, 100)
        """

        colors = {
            'Hufflepuff':'c',
            'Ravenclaw':'orange',
            'Slytherin':'g',
            'Gryffindor':'r',
        }

        plt.figure(figsize=(10, 5))

        for house in unique_houses:
            plt.hist(to_plot[house], bins, alpha=0.5, label=house, color=colors[house])

        plt.legend(loc='upper right')
        plt.title('Histogram of "%s" grades among the different Hogwarts houses' % self.data_set.data_set[0][col_nb])
        plt.show(block=True)



if __name__=='__main__':
    '''You have to run it with python3'''
    try:
        col_nb = int(sys.argv[1])
    except:
        col_nb = 16

    h = HistogramPerHouse()
    h.Plot(col_nb)

