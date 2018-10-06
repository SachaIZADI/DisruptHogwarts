import numpy as np
import os
import sys



class SplitTestTrain:

    def __init__(self, X, y, ratio=0.7):
        self.X = X
        self.y = y
        self.ratio = ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Split(self):
        index = [i for i in range(self.X.shape[0])]
        np.random.shuffle(index)

        self.X_train = self.X[index[:int(len(index) * self.ratio)],]
        self.y_train = self.y[index[:int(len(index) * self.ratio)],]

        self.X_test = self.X[index[int(len(index) * self.ratio):],]
        self.y_test = self.y[index[int(len(index) * self.ratio):],]




class ConfusionMatrix:

    def __init__(self, y_predict, y_true, unique_labels=None):
        self.y_predict = y_predict
        self.y_true = y_true
        if unique_labels is None:
            self.unique_labels = list(set(self.y_true))
        else:
            self.unique_labels = unique_labels
        self.confusion = None

    def getMatrix(self):
        self.confusion = np.zeros(shape=(len(self.unique_labels),len(self.unique_labels)),dtype=int)
        # confusion[('true','predicted')] = nb true labels predicted as predicted
        for i in range(self.y_predict.shape[0]):
            self.confusion[self.unique_labels.index(self.y_true[i]), self.unique_labels.index(self.y_predict[i])] += 1

        self.confusion = np.vstack((self.unique_labels, self.confusion))
        new_col = np.array([' ']+self.unique_labels)
        new_col.shape = (new_col.shape[0],1)
        self.confusion = np.hstack((new_col, self.confusion))
        return(self.confusion)


    def Print(self):
        cell_size = len(max(self.unique_labels, key=len))
        template = '{:<%ds}   ' % cell_size
        for i in range(self.confusion.shape[0]):
            to_print = ''
            for j in range(self.confusion.shape[1]):
                to_print += template.format(str(self.confusion[i,j]))
            to_print += '\n'
            sys.stdout.write(to_print)





#TODO: implement other metrics