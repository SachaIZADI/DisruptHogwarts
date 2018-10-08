import numpy as np
import sys



class SplitTrainTest:
    '''
    - Splits a dataset into a testing and training set
    - Example to run:
        from metrics import SplitTrainTest
        import numpy as np
        X = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
        y = np.array(['a','a','b','c','b'])
        sptt = SplitTrainTest(X,y,ratio=0.6)
        sptt.Split()
        print(sptt.X_train)
    '''

    def __init__(self, X, y, ratio=0.7):
        '''
        :param X: a 2D np.array. The feature matrix.
        :param y: a 1D np.array. The target variable
        :param ratio: a float between 0 and 1. The ratio between nb of training examples and total nb of examples
        '''
        self.X = X
        self.y = y
        self.ratio = ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Split(self):
        '''
        Executes the split.
        '''
        index = [i for i in range(self.X.shape[0])]
        np.random.shuffle(index)

        self.X_train = self.X[index[:int(len(index) * self.ratio)],]
        self.y_train = self.y[index[:int(len(index) * self.ratio)],]

        self.X_test = self.X[index[int(len(index) * self.ratio):],]
        self.y_test = self.y[index[int(len(index) * self.ratio):],]




class ConfusionMatrix:
    '''
    - Computes a confusion matrix.
    - Example to run:
        from metrics import ConfusionMatrix
        import numpy as np
        y_true = np.array(['a','a','b','c','b'])
        y_pred = np.array(['b','a','c','c','a'])
        cm = ConfusionMatrix(y_pred, y_true)
        cm.getMatrix()
    '''

    def __init__(self, y_predict, y_true, unique_labels=None):
        '''
        :param y_predict: a 1D np.array of strings. The predicted values.
        :param y_true: a 1D np.array of strings. The true values.
        :param unique_labels: a list of strings.
        '''
        self.y_predict = y_predict
        self.y_true = y_true
        if unique_labels is None:
            self.unique_labels = list(set(self.y_true))
        else:
            self.unique_labels = unique_labels
        self.confusion = None


    def getMatrix(self):
        '''
        Computes the confusion matrix.
        '''
        self.confusion = np.zeros(shape=(len(self.unique_labels),len(self.unique_labels)),dtype=int)
        # confusion[('true','predicted')] = nb true labels predicted as predicted
        for i in range(self.y_predict.shape[0]):
            self.confusion[self.unique_labels.index(self.y_true[i]), self.unique_labels.index(self.y_predict[i])] += 1

        self.confusion = np.vstack((self.unique_labels, self.confusion))
        new_col = np.array([' ']+self.unique_labels)
        new_col.shape = (new_col.shape[0],1)
        self.confusion = np.hstack((new_col, self.confusion))
        return self.confusion


    def Print(self):
        '''
        - Prints the confusion matrix to stdout in a fancy way.
        - To use only if the script is called from the shell, otherwise it will crash.
        '''
        cell_size = len(max(self.unique_labels, key=len))
        template = '{:<%ds}   ' % cell_size
        for i in range(self.confusion.shape[0]):
            to_print = ''
            for j in range(self.confusion.shape[1]):
                to_print += template.format(str(self.confusion[i,j]))
            to_print += '\n'
            sys.stdout.write(to_print)





#TODO: implement other metrics