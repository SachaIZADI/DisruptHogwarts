import sys
import os
from logistic_regression import LogisticRegression
from preprocessing import MeanImputation, Scaling
from describe import DataSet
from utils import convert_to_float
import numpy as np
from metrics import ConfusionMatrix, SplitTestTrain


def main():

    file_name = sys.argv[1]
    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, file_name)

    d = DataSet(file_name)
    d.loadDataSet()

    to_remove = [
        d.data_set[0].index('Index'),
        d.data_set[0].index('First Name'),
        d.data_set[0].index('Last Name'),
        d.data_set[0].index('Birthday'),
        d.data_set[0].index('Best Hand'),
        d.data_set[0].index('Hogwarts House'),
        # remove other features
    ]

    X = np.array([[d.data_set[i][j] for j in range(len(d.data_set[0])) if j not in to_remove]
                for i in range(len(d.data_set))])
    X = convert_to_float(X[1:,])

    y_col_nb = d.data_set[0].index('Hogwarts House')
    y = np.array(d.extractColumn(y_col_nb)[1:])

    m = MeanImputation(X)
    m.train()
    m.transform()

    sc = Scaling(X)
    sc.train()
    sc.transform()

    sp = SplitTestTrain(X, y)
    sp.Split()
    X_train = sp.X_train
    y_train = sp.y_train
    X_test = sp.X_test
    y_test = sp.y_test


    l = LogisticRegression(X=X_train, y=y_train)
    l.train()

    y_predicted = l.predict(X_test)

    cm = ConfusionMatrix(y_test, y_predicted)
    confusion_matrix = cm.getMatrix()

    print(confusion_matrix)
    #TODO: better print the confusion matrix


if __name__=='__main__':
    main()