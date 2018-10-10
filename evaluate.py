import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # Load the truths
    truths = pd.read_csv('resources/dataset_truth.csv', sep=',', index_col=0)
    # Load predictions
    predictions = pd.read_csv('resources/houses.csv', sep=',', index_col=0)
    # Replace names by numerical value {0, 1, 2, 3} and convert to array
    houses = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    y_true = truths.replace(houses).as_matrix()
    y_pred = predictions.replace(houses).as_matrix()
    # Print the score using accuracy_score
    print("Your score on test set: {}".format(accuracy_score(y_true.reshape((400, )),
                                              y_pred.reshape((400, )))))
