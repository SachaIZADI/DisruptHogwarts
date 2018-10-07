import numpy as np
import math
import json
import os
import random

class LogisticRegression:

    def __init__(self, X, y=None,
                 path_to_beta=None,
                 regularization=None, C=1,
                 optimizer='gradient_descent', optimizer_params={'alpha':0.5,'n':100}):
        self.X = X
        self.y = y
        if y is not None:
            self.unique_labels = list(set(y))
        self.regularization = regularization
        self.C = C
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if path_to_beta is None:
            self.beta = {}
            for label in self.unique_labels[:-1]:
                self.beta[label] = np.random.uniform(-5, 5, X.shape[1])
            self.beta[self.unique_labels[-1]] = np.array([0.0 for i in range(X.shape[1])])
        else:
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, path_to_beta)
            with open(file_name,'r') as f:
                self.beta = json.loads(f.read())
            self.unique_labels = [key for key in self.beta]

    def loss(self):
        # compute loss
        # http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
        m = self.X.shape[0]
        loss = 0
        for i in range(m):
            Z = 0
            for j in range(len(self.unique_labels)):
                Z += math.exp(np.dot(self.beta[self.unique_labels[j]], self.X[i]))
            loss += math.log(
                math.exp(np.dot(self.beta[self.y[i]], self.X[i])) / Z
            )
        loss = -1/m * loss

        if self.regularization == 'l2':
            for label in self.unique_labels:
                loss += self.C * np.dot(self.beta[label], self.beta[label])

        return loss



    def softmax(self, y):
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
            if self.regularization=='l2':
                full_gradient[label] += 2 * self.C * self.beta[label]
        return full_gradient



    def gradient_descent(self, show_progress=True):
        params = self.optimizer_params
        for i in range(params['n']):
            full_gradient = self.full_gradient()
            for label in self.unique_labels[:-1]:
                self.beta[label] = self.beta[label] - params['alpha']*full_gradient[label]

            if show_progress and i%20==0:
                print("iteration n°%s - - - - - - - - - - - loss: %s" % (i, self.loss()))
                # TODO : dynamic plotting of the optimization process
        print("iteration n°%s - - - - - - - - - - - loss: %s" % (i, self.loss()))


    def stochastic_gradient_descent(self,show_progress=True):
        params = self.optimizer_params
        for i in range(params['n']):
            m = self.X.shape[0]
            indexes = [i for i in range(m)]
            random.shuffle(indexes)
            b = params['batch_size']
            for k in range(m//b+1):
                Xk = self.X[indexes[k*b:max((k+1)*b, m)]]
                yk = self.y[indexes[k*b:max((k+1)*b, m)]]
                gradient = {}
                for label in self.unique_labels[-1]:
                    gradient[label] = 0
                for j in range(Xk.shape[0]):
                    grad = self.unit_gradient(Xk[j], yk[j])
                    for label in self.unique_labels:
                        gradient[label] += grad[label]

                for label in self.unique_labels[:-1]:
                    self.beta[label] = self.beta[label] - params['alpha'] * gradient[label]

                if show_progress and i % 20 == 0:
                    print("iteration n°%s - - - - - - - - - - - loss: %s" % (i, self.loss()))
                    # TODO : dynamic plotting of the optimization process
            print("iteration n°%s - - - - - - - - - - - loss: %s" % (i, self.loss()))



    def train(self):
        if self.optimizer == 'gradient_descent':
            self.gradient_descent()
        elif self.optimizer == 'sgd':
            self.stochastic_gradient_descent()
        #TODO : add other optimizers Adam, Newton-Raphson
        else:
            return

        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, 'results/beta.json')
        with open(file_name, 'w+') as outfile:
            beta_json = {}
            for label in self.unique_labels:
                beta_json[label] = list(self.beta[label])
            json.dump(beta_json, outfile)


    def predict(self, X_to_predict=None):
        if X_to_predict is None:
            X_to_predict = self.X
        prediction = []
        for x in X_to_predict:
            probas = self.probabilities(x)
            prediction += [max(probas, key=probas.get)]
        return prediction