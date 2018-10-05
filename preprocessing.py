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
        mean_imputation_dict = {}
        for j in range(self.X.shape[1]):
            feature = [x for x in self.X[:,j] if not np .isnan(x)]
            st = Statistics(feature)
            m = st.Mean()
            mean_imputation_dict[j] = m

        self.path_to_mean_imputation = 'results/mean_imputation_%s.json' % str(datetime.now()).replace(' ', '_').split('.')[0]
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, self.path_to_mean_imputation)
        with open(file_name, 'w+') as outfile:
            json.dump(mean_imputation_dict, outfile)


    def transform(self):
        if self.mean_imputation_dict:
            return
        else :
            
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, self.path_to_mean_imputation)
        with open(file_name, 'w+') as outfile:
            json.dump(mean_imputation_dict, outfile)
        return


class Scaling:

    def __init__(self, X, path_to_scaling=None):
        self.X = X
        self.path_to_means = path_to_scaling

    def train(self):
        return

    def transform(self):
        return