import matplotlib.pyplot as plt
from describe import DataSet


class HistogramPerHouse:

    def __init__(self, path_to_data_set='resources/dataset_train.csv'):
        self.data_set = DataSet(path_to_data_set)
        self.data_set.loadDataSet()

    def plot(self, col_nb):
        feature = self.data_set.extractColumn(col_nb=col_nb, convert_to_float=True)[1:]
        houses = self.data_set.extractColumn(col_nb=1)[1:]

        to_plot = {}

        for i in range(len(houses)):
            try :
                to_plot[houses[i]] += [feature[i]]
            except :
                to_plot[houses[i]] = [feature[i]]

        return to_plot