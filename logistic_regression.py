import numpy as np
import math

class LogisticRegression:

    def __init__(self, X, y=None, train=True, path_to_beta=None, optimizer='gradient_descent', optimizer_params={'alpha':0.5}):
        self.X = X
        self.y = y
        self.unique_labels = list(set(y))
        self.train = train
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if path_to_beta is None :
            self.beta = {}
            for label in self.unique_labels[:-1]:
                self.beta[label] = np.random.uniform(-5, 5, X.shape[1])
            self.beta[self.unique_labels[-1]] = np.array([0 for i in range(X.shape[1])])
        else:
            #ToDo: Load beta
            self.beta=None


    def imputation(self):
        #impute missing values
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



    def gradient(self):
        m = self.X.shape[0]
        gradient = {}

        for label in self.unique_labels:
            gradient[label] = 0
            for i in range(m):
                probas = self.probabilities(self.X[i])
                gradient[label] += self.X[i] * ((self.y[i]==label) - probas[label])
            gradient[label] = 1/m * gradient[label]

        return gradient


    def gradient_descent_step(self):
        return




    def train(self):
        return

    def predict(self):
        return

