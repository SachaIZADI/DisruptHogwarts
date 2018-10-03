import sys
import os
import csv
import math


class Statistics:
    """
    - Computes some statistics about a list of floats.
    - Missing values are ignored.
    - Example to run:
        from describe import Statistics
        list = [1,2,3,4,5]
        s = Statistics(list)
        mean = s.Mean()
        first_quartile = s.Quartile(0.25)
    """

    def __init__(self, list):
        self.list = list
        self.sorted = False


    def Count(self):
        list = [x for x in self.list if x]
        return len(list)


    def Sum(self):
        sum = 0
        for elt in self.list:
            sum += elt
        return(sum)


    def Mean(self):
        sum = self.Sum()
        n = self.Count()
        return sum/n


    def Std(self, unbiased=True):
        """
        :param unbiased: boolean.
        If True, the unbiased estimate of the std is computed (we divide by N-1).
        If False, the MLE of the std is computed (we divide by N)
        """
        list = [x for x in self.list if x]
        mean = self.Mean()
        centered_squared_list = [(elt-mean)**2 for elt in list]
        stats = Statistics(centered_squared_list)
        S2 = stats.Sum()
        N = stats.Count()
        if unbiased:
            return math.sqrt(S2/(N-1))
        else:
            return math.sqrt(S2/N)


    def Sort(self):
        """
        Sorts a list
        """
        # TODO: code our own sorting function
        if not self.sorted:
            self.list = [x for x in self.list if x]
            self.list.sort()
            self.sorted = True
        else:
            pass


    def Quartile(self, percentage):
        """
        :param percentage: float in [0,1]
        Computes the quartile in a list.
        NB:
            Min: percentage = 0
            1st quartile: percentage = .25
            Median: percentage = .5
            3rd quartile: percentage = .75
            Max: percentage = 1
        """
        self.Sort()
        N = self.Count() - 1
        if (percentage * N) % 1 == 0:
            return self.list[math.floor(percentage * N)]
        else:
            return (self.list[math.floor(percentage * N)] + self.list[math.floor(percentage * N)+1])/2



class DataSet:
    """
    - A useful class to load & handle csv datasets, and to compute statistics about it.
    - Example to run:
        from describe import DataSet
        d = DataSet("resources/dataset_train.csv", ',')
        d.loadDataSet()
        print(d.data_set[:3])
        d.computeStatistics()
        print(d.summary)
    - NB: printSummary() only works when called in the __main__ from a terminal.
    """

    def __init__(self, path_to_data_set, separator=','):
        """
        :param path_to_data_set: the path to your csv dataset. Can be either relative or absolute.
        :param separator: the separator in the csv (in the Hogwarts case, it is a ",").
        """
        self.path_to_data_set = path_to_data_set
        self.separator = separator
        self.data_set = []
        self.summary = None


    def loadDataSet(self):
        """
        Open and read the csv file. Store it in an array (actually a list of lists)
        """
        with open(self.path_to_data_set, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=self.separator)
            for row in csv_reader:
                self.data_set += [row]


    def extractColumn(self, col_nb, convert_to_float=False):
        """
        :param col_nb: an int. The position of the column you want to extract (starts at 0)
        :param convert_to_float: a boolean. Most likely, numbers are read by loadDataSet as characters, if convert_to_float is True, they will be converted to floats.
        :return: the whole col_nb-th column (i.e. feature) of the dataset
        """
        col = []
        ignore_line = True
        for row in self.data_set:
            if not convert_to_float or ignore_line:
                col += [row[col_nb]]
                ignore_line = False
            else:
                try:
                    col += [float(row[col_nb])]
                except Exception as e:
                    if str(e) == 'could not convert string to float: ':
                        # NB: if there is a missing value, we encode it by False
                        col += [False]
        return col


    def isNumericFeature(self, col_nb):
        """
        Automatically identifies if a feature/column is numeric or not.
        The method returns True if more than 90% of the values of the feature are numeric.
        The method doesn't handle dates data.
        :param col_nb: an int. The position of the column you want to test
        :return: boolean. True ,if the feature if numeric. False, if not.
        """
        feature = self.extractColumn(col_nb)
        proportion_numeric = 0
        proportion_empty = 0

        for elt in feature:
            if elt == '':
                proportion_empty += 1
            else:
                proportion_numeric += elt.replace('.', '', 1).replace('-', '', 1).isdigit()

        if len(self.data_set)-proportion_empty == 0:
            return False

        proportion_numeric = proportion_numeric / (len(self.data_set)-proportion_empty)

        if proportion_numeric > 0.9:
            return True
        else:
            return False



    def computeStatistics(self):
        """
        Computes some statistics about the numeric features in the dataset.
        """
        self.summary = []

        for col_nb in range(len(self.data_set[0])):
            if self.isNumericFeature(col_nb):
                column = self.extractColumn(col_nb, convert_to_float=True)
                stats = Statistics(column[1:])

                col_summary = {}
                col_summary['Feature'] = column[0]
                col_summary['Count'] = stats.Count()
                col_summary['Mean'] = stats.Mean()
                col_summary['Std'] = stats.Std()
                col_summary['Min'] = stats.Quartile(0)
                col_summary['25%'] = stats.Quartile(.25)
                col_summary['50%'] = stats.Quartile(.5)
                col_summary['75%'] = stats.Quartile(.75)
                col_summary['Max'] = stats.Quartile(1)

                self.summary += [col_summary]


    def printSummary(self):
        """
        Prints in a fancy way the statistics around the dataset.
        To be called in the __main__ from a terminal.
        The function is "responsive", i.e. it will be differently displayed depending on the size of the terminal window.
        """
        rows, columns = os.popen('stty size', 'r').read().split()
        columns = int(columns)
        cell_size = max([len(elt['Feature']) for elt in self.summary]) + 2
        template = '{:<%ds}| ' % cell_size
        columns_bis = (columns - cell_size - 1) // (cell_size+1)


        nb_features = len(self.summary)
        rows_bis = nb_features // columns_bis + 1


        for i in range(rows_bis-1):
            for row_name in ['Feature','Count','Mean','Std','Min','25%','50%','75%','Max']:
                to_print = template.format(row_name)

                for k in range(columns_bis):
                    if row_name == 'Feature':
                        to_print += template.format(str(self.summary[i*columns_bis+k][row_name]))
                    else:
                        to_print += template.format(str(
                            round(self.summary[i * columns_bis + k][row_name],3)
                        ))

                sys.stdout.write(to_print+'\n')

                if row_name == 'Feature':
                    line = ''
                    for z in range(int(columns)):
                        line += '-'
                    sys.stdout.write(line + '\n')

            sys.stdout.write('\n'+'\n')



if __name__=='__main__':
    '''
    - How to run it: python3 describe.py resources/dataset_train.csv
    - /!\ Make sure to use python3 and not python2 /!\ 
    '''

    file_name = sys.argv[1]
    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, file_name)

    d = DataSet(file_name)
    d.loadDataSet()
    d.computeStatistics()
    d.printSummary()
