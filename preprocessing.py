import numpy as np
from describe import Statistics
import os
from datetime import datetime
import json


class MeanImputation:

    def __init__(self, X, path_to_mean_imputation=None):
        self.X = X
        self.path_to_mean_imputation = path_to_mean_imputation
        self.mean_imputation_dict = None


    def train(self):
        self.mean_imputation_dict = {}
        for j in range(self.X.shape[1]):
            feature = [x for x in self.X[:,j] if not np .isnan(x)]
            st = Statistics(feature)
            m = st.Mean()
            self.mean_imputation_dict[j] = m

        self.path_to_mean_imputation = 'results/mean_imputation_%s.json' % str(datetime.now()).replace(' ', '_').split('.')[0]
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, self.path_to_mean_imputation)
        with open(file_name, 'w+') as outfile:
            json.dump(self.mean_imputation_dict, outfile)


    def transform(self):
        if not self.mean_imputation_dict:
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, self.path_to_mean_imputation)
            with open(file_name, 'r') as f:
                self.mean_imputation_dict = json.loads(f.read())

        for j in range(self.X.shape[1]):
            for i in range(self.X.shape[0]):
                if np.isnan(self.X[i,j]):
                    self.X[i,j] = self.mean_imputation_dict[j]




class Scaling:

    def __init__(self, X, path_to_scaling=None):
        self.X = X
        self.path_to_means = path_to_scaling
        self.mean_dict = None
        self.std_dict = None

    def train(self):
        self.mean_dict = {}
        self.std_dict = {}
        return

    def transform(self):
        return