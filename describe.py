import csv
import math
import sys
import os


class Statistics:

    def __init__(self, list):
        self.list = list
        self.sorted = False


    def Count(self):
        return len(self.list)


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
        mean = self.Mean()
        centered_squared_list = [(elt-mean)**2 for elt in self.list]
        stats = Statistics(centered_squared_list)
        S2 = stats.Sum()
        N = stats.Count()
        if unbiased:
            return math.sqrt(S2/(N-1))
        else:
            return math.sqrt(S2/N)


    def Sort(self):
        if not self.sorted :
            self.list.sort()
            self.sorted = True
        else:
            pass


    def Max(self):
        self.Sort()
        return self.list[-1]


    def Min(self):
        self.Sort()
        return self.list[0]


    def Quartile(self, percentage):
        self.Sort()
        N = self.Count() - 1
        if (percentage * N) % 1 == 0:
            return self.list[math.floor(percentage * N)]
        else:
            return (self.list[math.floor(percentage * N)] + self.list[math.floor(percentage * N)+1])/2





class DataSet:

    def __init__(self, path_to_data_set, separator=','):
        self.path_to_data_set = path_to_data_set
        self.separator = separator
        self.data_set = []
        self.summary = None


    def loadDataSet(self):
        with open(self.path_to_data_set, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=self.separator)
            for row in csv_reader:
                self.data_set += [row]


    def extractColumn(self, col_nb, convert_to_float=False):
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
                        col += [False]

        return col


    def isNumericFeature(self, col_nb):
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

        self.summary = []

        for col_nb in range(len(self.data_set[0])):
            if self.isNumericFeature(col_nb):

                # TODO: Code is broken here

                column = self.extractColumn(col_nb, convert_to_float=True)
                stats = Statistics(column[1:])

                col_summary = {}
                col_summary['col_name'] = column[0]
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
        # TODO: improve the print function

        n = len(self.summary) + 1

        template = ''

        for i in range(n) :
            template += '%s \t '

        col_names = tuple([' '] + [elt['col_name'] for elt in self.summary])
        Count = tuple(['Count'] + [elt['Count'] for elt in self.summary])
        Mean = tuple(['Mean'] + [elt['Mean'] for elt in self.summary])
        Std = tuple(['Std'] + [elt['Std'] for elt in self.summary])
        Min = tuple(['Min'] + [elt['Min'] for elt in self.summary])
        fst_quartile = tuple(['25%'] + [elt['25%'] for elt in self.summary])
        scd_quartile = tuple(['50%'] + [elt['50%'] for elt in self.summary])
        trd_quartile = tuple(['75%'] + [elt['75%'] for elt in self.summary])
        Max = tuple(['Max'] + [elt['Max'] for elt in self.summary])


        to_print = \
            template % col_names + '\n' + \
            template % Count + '\n' + \
            template % Mean + '\n' + \
            template % Std + '\n' + \
            template % Min + '\n' + \
            template % fst_quartile + '\n' + \
            template % scd_quartile + '\n' + \
            template % trd_quartile + '\n' + \
            template % Max + '\n'

        return(to_print)



if __name__ == '__main__' :

    '''
    You have to run python3
    '''

    file_name = sys.argv[1]
    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, file_name)

    d = DataSet(file_name)
    d.loadDataSet()
    d.computeStatistics()
    sys.stdout.write(d.printSummary())