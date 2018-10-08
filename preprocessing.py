import numpy as np
from describe import Statistics
import os
import json


class MeanImputation:
    '''
    - Handles missing values by imputing the mean of the feature.
    - Imputing the mean is not, statistically-speaking, the best thing to do, but it's very simple to implement
    - Example to run :
        from preprocessing import MeanImputation
        import numpy as np
        X = np.array([[1,2,3],[1,np.nan,3],[1,2,np.nan]])
        m = MeanImputation(X)
        m.train()
        m.transform()
        print(m.X)
    '''

    def __init__(self, X, path_to_mean_imputation=None):
        '''
        :param X: a np.array of floats. The feature matrix.
        :param path_to_mean_imputation: a string. The path to a json of means.
        '''
        self.X = X
        self.path_to_mean_imputation = path_to_mean_imputation
        self.mean_imputation_dict = None


    def train(self):
        '''
        Computes the means of each feature of X.
        '''
        self.mean_imputation_dict = {}
        for j in range(self.X.shape[1]):
            feature = [x for x in self.X[:,j] if not np.isnan(x)]
            st = Statistics(feature)
            m = st.Mean()
            self.mean_imputation_dict[j] = m

        # Saves the means in a json file
        self.path_to_mean_imputation = 'results/mean_imputation.json'
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, self.path_to_mean_imputation)
        with open(file_name, 'w+') as outfile:
            json.dump(self.mean_imputation_dict, outfile)


    def transform(self):
        '''
        Fills-in each missing value by the imputed value
        '''
        # if a path_to_mean_imputation is specified, loads the mean imputation from it.
        loading_csv = False
        if not self.mean_imputation_dict:
            loading_csv = True
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, self.path_to_mean_imputation)
            with open(file_name, 'r') as f:
                self.mean_imputation_dict = json.loads(f.read())

        for j in range(self.X.shape[1]):
            for i in range(self.X.shape[0]):
                if np.isnan(self.X[i,j]):
                    if loading_csv:
                        self.X[i,j] = self.mean_imputation_dict[str(j)]
                    else:
                        self.X[i, j] = self.mean_imputation_dict[j]




class Scaling:
    '''
    - Centers and scales all features in X: X[,j] = (X[,j]-µ_j)/σ_j
    - Example to run:
        from preprocessing import Scaling
        import numpy as np
        X = np.array([[1,2,3],[1.1,2.1,3.1],[0.9,1.9,2.9]])
        sc = Scaling(X)
        sc.train()
        sc.transform()
        print(sc.X)
    '''

    def __init__(self, X, path_to_scaling=None):
        '''
        :param X: a np.array of floats. The feature matrix.
        :param path_to_scaling: a string. The path to a json of (µ_j,σ_j).
        '''
        self.X = X
        self.path_to_scaling = path_to_scaling
        self.mean_dict = None
        self.std_dict = None


    def train(self):
        '''
        Computes the mean and the standard deviation of each feature
        '''
        self.mean_dict = {}
        self.std_dict = {}

        for j in range(self.X.shape[1]):
            feature = [x for x in self.X[:,j]]
            st = Statistics(feature)
            m = st.Mean()
            std = st.Std()
            self.mean_dict[j] = m
            self.std_dict[j] = std

        # Saves the means and std's to a json file
        self.path_to_scaling = 'results/scaling.json'
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, self.path_to_scaling)
        with open(file_name, 'w+') as outfile:
            json.dump({'mean':self.mean_dict,
                       'std':self.std_dict},
                      outfile)


    def transform(self):
        '''
        Centers and scales the features
        '''
        # if a path_to_scaling is specified, loads the means and stds from it.
        loading_csv = False
        if not self.mean_dict:
            loading_csv = True
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, self.path_to_scaling)
            with open(file_name, 'r') as f:
                scaling = json.loads(f.read())
            self.mean_dict = scaling['mean']
            self.std_dict = scaling['std']

        for j in range(self.X.shape[1]):
            if loading_csv:
                self.X[:,j] = (self.X[:,j]-self.mean_dict[str(j)])/self.std_dict[str(j)]
            else:
                self.X[:, j] = (self.X[:, j] - self.mean_dict[j]) / self.std_dict[j]