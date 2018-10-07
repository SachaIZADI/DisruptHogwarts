import sys
import os
from logistic_regression import LogisticRegression
from preprocessing import MeanImputation, Scaling
from describe import DataSet
from utils import convert_to_float
import numpy as np
import csv


def main():

    dirname = os.path.dirname(__file__)
    dirname_prediction = os.path.join(dirname, 'results')

    file_name = sys.argv[1]
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

    index_position = d.data_set[0].index('Index')
    indexes = np.array([d.data_set[i][index_position]for i in range(len(d.data_set))])[1:]

    X = np.array([[d.data_set[i][j] for j in range(len(d.data_set[0])) if j not in to_remove]
                for i in range(len(d.data_set))])
    #features = X[0,:]
    X = convert_to_float(X[1:,])

    m = MeanImputation(X, path_to_mean_imputation=os.path.join(dirname_prediction, 'mean_imputation.json'))
    m.transform()

    sc = Scaling(X, path_to_scaling=os.path.join(dirname_prediction, 'scaling.json'))
    sc.transform()

    l = LogisticRegression(X=X, path_to_beta=os.path.join(dirname_prediction, 'beta.json'))
    predictions = l.predict()



    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, 'resources/houses.csv')
    with open(file_name, 'w+') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Index', 'Hogwarts House'])
        for i in range(len(indexes)):
            writer.writerow([indexes[i], predictions[i]])



if __name__=='__main__':
    main()