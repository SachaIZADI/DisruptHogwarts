import numpy as np
import math
import json
import os
import random

class LogisticRegression:
    '''
    - A class more or less similar to scikit's logistic regression
    - How to use it :
        from logistic_regression import LogisticRegression
        import numpy as np
        X = np.array([[1,2,3],[2,3,4],[3,4,5]])
        y = np.array(['a','b','a'])
        # Instantiate a LogisticRegression object
        l = LogisticRegression(X,y)
        # Train the model
        l.train()
        # Make predictions
        x = np.array([3,2,1])
        l.predict(x)
    '''


    def __init__(self, X, y=None,
                 path_to_beta=None,
                 regularization=None, C=1,
                 optimizer='gradient_descent', optimizer_params={'alpha':0.5,'n':100}):
        '''
        :param X: a 2D np.array of floats. The feature matrix.
        :param y: a 1D np.array of STRINGS. The associated labels.
        :param path_to_beta: a string. The path to a JSON file containing the values of beta with which you want to make the prediction.
        :param regularization: None or 'l2'. If l2 is specified, the algorithm will perform a l2 regularization.
        :param C: a positive float. The hyperparameter of the l2 regularization.
        :param optimizer: 'gradient_descent' or 'sgd'. Chose one of the 2 optimizers implemented.
        :param optimizer_params: a dictionary. The values of the hyperparameters used to run the optimizers.
                    if optimizer is 'gradient_descent' : optimizer_params={'alpha':0.5,'n':100}, where 'alpha' is the learning rate and 'n' is the number of iterations.
                    if optimizer is 'sgd' : optimizer_params={'alpha':0.5,'n':5,'batch_size':16}, where ... idem ... and 'batch_size' is the size of the mini batches used to compute the estimate of the gradient.
        '''
        self.X = X
        self.y = y
        if y is not None:
            self.unique_labels = list(set(y))
        self.regularization = regularization
        self.C = C
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if path_to_beta is None:
            # if no path_to_beta is specified, we initialize beta randomly.
            self.beta = {}
            for label in self.unique_labels[:-1]:
                self.beta[label] = np.random.uniform(-5, 5, X.shape[1])
            # NB : due to model identification issues, we 'stabilize' the algorithm by standardizing the last value of beta to [0,0,...,0]^T
            self.beta[self.unique_labels[-1]] = np.array([0.0 for i in range(X.shape[1])])
        else:
            # if path_to_beta is specified, we load the values of beta from the json file.
            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, path_to_beta)
            with open(file_name,'r') as f:
                self.beta = json.loads(f.read())
            self.unique_labels = [key for key in self.beta]



    def loss(self):
        '''
        - Compute the loss of the model. Loss = - log-likelihood
        - More details can be found here: http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
        '''
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
        '''
        Compute the softmax of an array y.
        :param y: a np.array of floats.
        '''
        Z = 0
        for i in range(len(y)):
            Z += y[i]
        return y/Z



    def probabilities(self, x):
        '''
        Computes the probabilities P(y=j|X=x) with the current value of beta.
        :param x: a 1D np.array of floats.
        '''
        probas = []
        for i in range(len(self.unique_labels)):
            probas += [math.exp(np.dot(self.beta[self.unique_labels[i]], x))]
        probas = self.softmax(np.array(probas))

        probas_labels = {}
        for i in range(len(self.unique_labels)):
            probas_labels[self.unique_labels[i]] = probas[i]

        return probas_labels



    def unit_gradient(self, x, y):
        '''
        Computes the gradient contribution of a single individual: ∂L(x_i,y_i)/∂ß
        :param x: a 1D np.array of floats.
        :param y: a string.
        '''
        gradient = {}
        for label in self.unique_labels:
            probas = self.probabilities(x)
            gradient[label] = x * ((y == label) - probas[label])
        return gradient


    def full_gradient(self):
        '''
        Computes the full gradient, averaging the contributions of all individuals in the full training set: 1/m ∑ ∂L(x_i,y_i)/∂ß
        '''
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
        '''
        Runs the gradient descent algorithm (using the full gradient over the full training set).
        :param show_progress: a boolean. If True, will print regularly the progress of the learning algorithm.
        '''
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
        '''
        Runs the stochastic gradient descent algorithm (computing an estimate of the gradient over minibatches of individuals randomly sampled from the training set).
        :param show_progress: a boolean. If True, will print regularly the progress of the learning algorithm.
        '''
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
                for label in self.unique_labels[:-1]:
                    gradient[label] = 0
                for j in range(Xk.shape[0]):
                    grad = self.unit_gradient(Xk[j], yk[j])
                    for label in self.unique_labels[:-1]:
                        gradient[label] += -(1/Xk.shape[0]) * grad[label]

                for label in self.unique_labels[:-1]:
                    self.beta[label] = self.beta[label] - params['alpha'] * gradient[label]

                if show_progress and k%10==0 :
                    print("iteration n°%s / batch n°%s - - - - - - - - - - - loss: %s" % (i, k, self.loss()))
                # TODO : dynamic plotting of the optimization process
        print("End of the training phase - - - - - - - - - - - loss: %s" % self.loss())



    def train(self):
        '''
        Executes the training depending on the choice of optimizer specified when initializing the model.
        '''
        if self.optimizer == 'gradient_descent':
            self.gradient_descent()
        elif self.optimizer == 'sgd':
            self.stochastic_gradient_descent()
        #TODO : add other optimizers Adam, Newton-Raphson
        else:
            return

        # Beta is automatically saved in a json file in results
        dirname = os.path.dirname(__file__)
        file_name = os.path.join(dirname, 'results/beta.json')
        with open(file_name, 'w+') as outfile:
            beta_json = {}
            for label in self.unique_labels:
                beta_json[label] = list(self.beta[label])
            json.dump(beta_json, outfile)


    def predict(self, X_to_predict=None):
        '''
        Infering values on new, unknown examples.
        :param X_to_predict: an 2D np.array of new examples (X_to_predict.shape[1] = X.shape[1]). If X_to_predict is None, the prediction is made over the training features matrix.
        '''
        if X_to_predict is None:
            X_to_predict = self.X
        prediction = []
        for x in X_to_predict:
            probas = self.probabilities(x)
            prediction += [max(probas, key=probas.get)]
        return prediction