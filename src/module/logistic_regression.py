import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression():
    def __init__(self,X_train, y_train):
        '''
        X - np.array(m,n) features matrix of m examples and n features
        Y - np.array(m,) labels of m examples
        w - np.array(n,) weights of n features
        b - scalar bias value
        '''
        self.weights, self.bias = None, None
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        self.X, self.y = self.zscore_normalize(X_train), y_train

    def zscore_normalize(self, X):
        return (X-self.mean)/self.std

    def _yhat(self, X, w, b):
        '''
        Returns predicted y (yhat)
        '''
        return np.dot(X,w) + b
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute_cost(self, y_true, y_predict):
        '''
        Returns Binary Cross entropy
        '''
        expr = y_true * np.log(y_predict) + (1 - y_true) * np.log(1 - y_predict)
        return -np.mean(expr)


    def fit(self, lr):
        n_samples, n_features = self.X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(1000):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(self.X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(self.X.T, (y_predicted - self.y))
            db = (1 / n_samples) * np.sum(y_predicted - self.y)
            # update parameters
            self.weights -= lr * dw
            self.bias -= lr * db

    def predict(self, X):
        '''
        Returns the binary classfied categories of the given X
        '''
        linear_model = np.dot(self.zscore_normalize(X), self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def MSE(self, y_predict, y_true):
        return np.mean((y_predict - y_true)**2)



if __name__=='__main__':

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12
    )

    classifier = LogisticRegression(X_train, y_train)
    classifier.fit(0.1)
    predictions = classifier.predict(X_test)
    print(accuracy(y_test, predictions))

