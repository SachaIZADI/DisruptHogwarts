import numpy as np


class SplitTestTrain:

    def __init__(self, X, y, ratio=0.7):
        self.X = X
        self.y = y
        self.ratio = ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def split(self):
        index = [i for i in range(self.X.shape[0])]
        np.random.shuffle(index)

        self.X_train = self.X[index[:int(len(index) * self.ratio)],]
        self.y_train = self.y[index[:int(len(index) * self.ratio)],]

        self.X_test = self.X[index[int(len(index) * self.ratio):],]
        self.y_test = self.y[index[int(len(index) * self.ratio):],]




