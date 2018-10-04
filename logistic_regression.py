import numpy as np
import math
import json
import os

class LogisticRegression:

    def __init__(self, X, y=None, path_to_beta=None, optimizer='gradient_descent', optimizer_params={'alpha':0.5,'n':100}):
        self.X = X
        self.y = y
        if y is not None:
            self.unique_labels = list(set(y))
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if path_to_beta is None :
            self.beta = {}
            for label in self.unique_labels[:-1]:
                self.beta[label] = np.random.uniform(-5, 5, X.shape[1])
            self.beta[self.unique_labels[-1]] = np.array([0 for i in range(X.shape[1])])
        else:
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, path_to_beta)
            with open(file_name,'r') as f:
                self.beta = json.loads(f.read())
            self.unique_labels = [key for key in self.beta]



    def imputation(self):
        #impute missing values
        return

    def preprocessing(self):
        #PCA / scaling
        return


    def loss(self):
        # compute loss
        # http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
        m = self.X.shape[0]
        loss = 0
        Z = 0
        for i in range(m):
            for j in range(len(self.unique_labels)):
                Z += math.exp(np.dot(self.beta[self.unique_labels[j]], self.X[i]))
            loss += math.log(
                math.exp(np.dot(self.beta[self.y[i]], self.X[i])) / Z
            )
        loss = -1/m * loss
        return loss



    def softmax(self,y):
        Z = 0
        for i in range(len(y)):
            Z += y[i]
        return y/Z



    def probabilities(self, x):
        probas = []
        for i in range(len(self.unique_labels)):
            probas += [math.exp(np.dot(self.beta[self.unique_labels[i]], x))]
        probas = self.softmax(np.array(probas))

        probas_labels = {}
        for i in range(len(self.unique_labels)):
            probas_labels[self.unique_labels[i]] = probas[i]

        return probas_labels



    def unit_gradient(self, x, y):
        gradient = {}
        for label in self.unique_labels:
            probas = self.probabilities(x)
            gradient[label] = x * ((y == label) - probas[label])
        return gradient


    def full_gradient(self):
        m = self.X.shape[0]
        full_gradient = {}
        for label in self.unique_labels:
            full_gradient[label] = 0
        for i in range(m):
            unit_gradient = self.unit_gradient(self.X[i], self.y[i])
            for label in self.unique_labels:
                full_gradient[label] += unit_gradient[label]
        for label in self.unique_labels:
            full_gradient[label] = -1/m * full_gradient[label]
        return full_gradient



    def gradient_descent(self, show_progress=True):
        params = self.optimizer_params
        for i in range(params['n']):
            for label in self.unique_labels:
                full_gradient = self.full_gradient()
                self.beta[label] = self.beta[label] - params['alpha']*full_gradient[label]

            if show_progress and i%20==0:
                print("iteration n°%s - - - - - - - - - - - loss: %s" % (i, self.loss()))
                # TODO : dynamic plotting of the optimization process



    def train(self):
        if self.optimizer == 'gradient_descent':
            self.gradient_descent()
        #TODO : add other optimizers, SGD, Adam, Newton-Raphson
        else:
            return


    def predict(self,x):
        probas = self.probabilities(x)
        return max(probas, key=probas.get)

