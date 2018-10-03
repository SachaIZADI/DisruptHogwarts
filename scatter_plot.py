import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from describe import DataSet, Statistics


class ScatterPlotPerHouse:

    def __init__(self, path_to_data_set='resources/dataset_train.csv', legend=True, size=10):
        self.data_set = DataSet(path_to_data_set)
        self.data_set.loadDataSet()
        self.legend = legend
        self.size = size

    def Plot(self, col_nb_1, col_nb_2):
        feature_1 = self.data_set.extractColumn(col_nb=col_nb_1, convert_to_float=True)[1:]
        feature_2 = self.data_set.extractColumn(col_nb=col_nb_2, convert_to_float=True)[1:]
        houses = self.data_set.extractColumn(col_nb=1)[1:]

        to_plot = {
                    'feature_1':{},
                    'feature_2':{},
                   }

        for i in range(len(houses)):
            if feature_1[i] and feature_2[i] :
                try:
                    to_plot['feature_1'][houses[i]] += [feature_1[i]]
                except:
                    to_plot['feature_1'][houses[i]] = [feature_1[i]]

                try:
                    to_plot['feature_2'][houses[i]] += [feature_2[i]]
                except:
                    to_plot['feature_2'][houses[i]] = [feature_2[i]]


        unique_houses = set(houses)
        colors = {
            'Hufflepuff':'c',
            'Ravenclaw':'orange',
            'Slytherin':'g',
            'Gryffindor':'r',
        }


        for house in unique_houses:
            plt.scatter(x=to_plot['feature_2'][house],
                        y=to_plot['feature_1'][house],
                        c=colors[house],
                        alpha=0.5,
                        label=house,
                        s=self.size)

        if self.legend:
            plt.legend(loc='upper right')
            plt.title('Scatter plot of "%s" vs "%s" grades among the different Hogwarts houses'
                      % (self.data_set.data_set[0][col_nb_1],self.data_set.data_set[0][col_nb_2]))
            plt.xlabel(self.data_set.data_set[0][col_nb_1])
            plt.ylabel(self.data_set.data_set[0][col_nb_2])



if __name__=='__main__':
    '''You have to run it with python3'''
    try:
        col_nb_1 = int(sys.argv[1])
        col_nb_2 = int(sys.argv[2])
    except:
        col_nb_1 = 6
        col_nb_2 = 16

    plt.figure(figsize=(10, 5))
    sc = ScatterPlotPerHouse()
    sc.Plot(col_nb_1, col_nb_2)
    plt.show(block=True)