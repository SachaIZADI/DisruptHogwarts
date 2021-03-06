import sys
import os
from logistic_regression import LogisticRegression
from preprocessing import MeanImputation, Scaling
from describe import DataSet
from utils import convert_to_float
import numpy as np
from metrics import ConfusionMatrix, SplitTrainTest


def main():
    '''
    Use this script to run experiments and fine-tune the algoritms
    '''

    # Load the dataset
    file_name = sys.argv[1]
    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, file_name)

    d = DataSet(file_name)
    d.loadDataSet()

    # Remove useless features (not numeric + bad regressors).
    to_remove = [
        d.data_set[0].index('Index'),
        d.data_set[0].index('First Name'),
        d.data_set[0].index('Last Name'),
        d.data_set[0].index('Birthday'),
        d.data_set[0].index('Best Hand'),
        d.data_set[0].index('Hogwarts House'),

        # Tests 7/10/18
        d.data_set[0].index('Arithmancy'),
        d.data_set[0].index('Defense Against the Dark Arts'),
        d.data_set[0].index('Divination'),
        d.data_set[0].index('Muggle Studies'),
        d.data_set[0].index('History of Magic'),
        d.data_set[0].index('Transfiguration'),
        d.data_set[0].index('Potions'),
        d.data_set[0].index('Care of Magical Creatures'),
        d.data_set[0].index('Charms'),
        d.data_set[0].index('Flying'),
    ]

    X = np.array([[d.data_set[i][j] for j in range(len(d.data_set[0])) if j not in to_remove]
                for i in range(len(d.data_set))])
    X = convert_to_float(X[1:,])

    y_col_nb = d.data_set[0].index('Hogwarts House')
    y = np.array(d.extractColumn(y_col_nb)[1:])

    # Impute missing values
    m = MeanImputation(X)
    m.train()
    m.transform()

    # Scale the variables
    sc = Scaling(X)
    sc.train()
    sc.transform()

    # Split the dataset in a training and testing set
    sp = SplitTrainTest(X, y)
    sp.Split()
    X_train = sp.X_train
    y_train = sp.y_train
    X_test = sp.X_test
    y_test = sp.y_test

    # Train a logistic regression model
    l = LogisticRegression(X=X_train, y=y_train)
    l.train()

    # Compute the confusion matrix over the training set
    y_predicted = l.predict()

    cm1 = ConfusionMatrix(y_train, y_predicted)
    cm1.getMatrix()
    print('\n\n')
    print('**************** Confusion Matrix on the training set ****************')
    print('\n')
    cm1.Print()

    # Compute the confusion matrix over the testing set
    y_predicted = l.predict(X_test)

    cm2 = ConfusionMatrix(y_test, y_predicted, cm1.unique_labels)
    cm2.getMatrix()
    print('\n\n')
    print('**************** Confusion Matrix on the testing set ****************')
    print('\n')
    cm2.Print()


if __name__=='__main__':
    '''
    - How to run it:
        python3 logreg_fine_tune.py resources/dataset_train.csv
    - /!\ Make sure to use python3 and not python2 /!\ 
    '''
    main()