import sys
import matplotlib.pyplot as plt
import numpy as np
from describe import DataSet, Statistics


class HistogramPerHouse:
    """
    - A class to plot the histogram of the Hogwarts features
    - Example to run:
        from histogram import HistogramPerHouse
        import matplotlib.pyplot as plt
        h = HistogramPerHouse()
        h.Plot(8)
        plt.show()
    """

    def __init__(self, path_to_data_set='resources/dataset_train.csv', legend=True, granularity=100):
        """
        :param path_to_data_set: a string. The path to the dataset.
        :param legend: a boolean. If legend is False, only the histogram is plotted. If legend is True, titles, axis legend, etc. are plotted.
        :param granularity: an integer. The number of barplots in the histogram.
        """
        self.data_set = DataSet(path_to_data_set)
        self.data_set.loadDataSet()
        self.legend = legend
        self.granularity = granularity


    def Plot(self, col_nb):
        """
        The plotting function.
        :param col_nb: integer. The position of the column / feature to plot.
        """
        feature = self.data_set.extractColumn(col_nb=col_nb, convert_to_float=True)[1:]
        houses = self.data_set.extractColumn(col_nb=1)[1:]

        to_plot = {}

        for i in range(len(houses)):
            try:
                to_plot[houses[i]] += [feature[i]]
            except:
                to_plot[houses[i]] = [feature[i]]


        full_list = []
        unique_houses = set(houses)
        for house in unique_houses:
            full_list += to_plot[house]

        s = Statistics(full_list)
        min = s.Quartile(0)
        max = s.Quartile(1)
        bins = np.linspace(min, max, self.granularity)

        colors = {
            'Hufflepuff':'c',
            'Ravenclaw':'orange',
            'Slytherin':'g',
            'Gryffindor':'r',
        }

        for house in unique_houses:
            plt.hist(to_plot[house], bins, alpha=0.5, label=house, color=colors[house])

        if self.legend :
            plt.legend(loc='upper right')
            plt.title('Histogram of "%s" grades among the different Hogwarts houses' % self.data_set.data_set[0][col_nb])
            plt.xlabel("Grade")
            plt.ylabel("Count")



if __name__=='__main__':
    '''
    - How to run it: 
        python3 histogram.py
        python3 histogram.py 9
    - /!\ Make sure to use python3 and not python2 /!\ 
    '''
    try:
        col_nb = int(sys.argv[1])
    except:
        col_nb = 16

    plt.figure(figsize=(10, 5))
    h = HistogramPerHouse()
    h.Plot(col_nb)
    plt.show(block=True)


