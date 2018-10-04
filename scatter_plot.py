import sys
import matplotlib.pyplot as plt
from describe import DataSet


class ScatterPlotPerHouse:
    """
    - A class to plot the scatter plot of a Hogwarts feature vs. another one
    - Example to run:
        from scatter_plot import ScatterPlotPerHouse
        import matplotlib.pyplot as plt
        sc = ScatterPlotPerHouse()
        sc.Plot(8,9)
        plt.show()
    """

    def __init__(self, path_to_data_set='resources/dataset_train.csv', legend=True, size=10):
        """
        :param path_to_data_set: a string. The path to the dataset.
        :param legend: a boolean. If legend is False, only the histogram is plotted. If legend is True, titles, axis legend, etc. are plotted.
        :param size: an int. The size of the points in the scatter plot.
        """
        self.data_set = DataSet(path_to_data_set)
        self.data_set.loadDataSet()
        self.legend = legend
        self.size = size

    def Plot(self, col_nb_1, col_nb_2):
        """
        Plotting function
        :param col_nb_1: integer. The position of the 1st column / feature to plot.
        :param col_nb_2: integer. The position of the 2nd column / feature to plot.
        """
        feature_1 = self.data_set.extractColumn(col_nb=col_nb_1, convert_to_float=True)[1:]
        feature_2 = self.data_set.extractColumn(col_nb=col_nb_2, convert_to_float=True)[1:]
        houses = self.data_set.extractColumn(col_nb=1)[1:]

        to_plot = {
                    'feature_1':{},
                    'feature_2':{},
                   }

        for i in range(len(houses)):
            if feature_1[i] and feature_2[i]:
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
            plt.scatter(x=to_plot['feature_1'][house],
                        y=to_plot['feature_2'][house],
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
    '''
    - How to run it: 
        python3 scatter_plot.py
        python3 scatter_plot.py 9 10
    - /!\ Make sure to use python3 and not python2 /!\ 
    '''
    try:
        col_nb_1 = int(sys.argv[1])
        col_nb_2 = int(sys.argv[2])
    except:
        col_nb_1 = 7
        col_nb_2 = 9

    plt.figure(figsize=(10, 5))
    sc = ScatterPlotPerHouse()
    sc.Plot(col_nb_1, col_nb_2)
    plt.show(block=True)