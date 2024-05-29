import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math

class SimpleLinearRegression():
    def __init__(self, file):
        self.file = file
        values = np.array(pd.read_csv(file).values,'float')
        self.X = values[:,0]
        self.Y = values[:,1]
        #Mean Normalization / Feature Scaling
        self.X = (self.X-np.mean(self.X))/(np.max(self.X)-np.min(self.X)) 
        self.w = 0
        self.b = 0
        self.true_w, self.true_b = 0.6872516865369586, 3.330238095238095
        self.cost_history = []

    def compute_cost(self):
        return 0.5 * (1/len(self.Y)) * self.OLS()
    
    def OLS(self):
        return np.sum((self.predict() - self.Y)**2)
    
    def predict(self):
        return np.dot(self.w, self.X) + self.b
    
    def compute_cost_manual(self, w, b):
        return 0.5 * (1/len(self.Y)) * np.sum((np.dot(w, self.X) + b - self.Y)**2)

    def fit(self, alpha, tolerance=1e-8, plot=None):
        last_cost = math.inf
        for i in range(100000):
            tmp_w = self.w - (alpha * (1/len(self.Y)) * np.sum(np.dot(self.predict() - self.Y, self.X)))
            tmp_b = self.b - (alpha * (1/len(self.Y)) * np.sum(self.predict() - self.Y))
            self.w = tmp_w
            self.b = tmp_b
            if(abs(last_cost-self.compute_cost())<tolerance):
                break
            self.cost_history.append(self.compute_cost())
            print('Epoch: {}, w: {}, b: {}, cost: {}'.format(i, self.w, self.b, self.compute_cost()))
            if(plot):
                if i == 0:
                    plt.figure(figsize=(16, 9))
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.scatter(self.X, self.Y, color='blue', label='Data Points')
                plt.plot(self.X, self.predict(), color='red')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Simple Linear Regression')
                plt.legend()
                plt.grid(True)

                plt.subplot(1,3,2)
                plt.plot(range(len(self.cost_history)), self.cost_history, color='green')
                plt.xlabel('Epochs')
                plt.ylabel('Cost')
                plt.title('Cost Function')
                plt.grid(True)

                plt.subplot(1,3,3)
                w_range = np.linspace(self.true_w - 0.8, self.true_w + 0.8, 10)
                b_range = np.linspace(self.true_b - 0.2, self.true_b + 0.2, 10)
                W, B = np.meshgrid(w_range, b_range)
                Cost = np.zeros_like(W)

                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        Cost[i, j] = self.compute_cost_manual(W[i, j], B[i, j])

                ax = plt.gcf().add_subplot(1, 3, 3, projection='3d')
                ax.plot_surface(W, B, Cost, cmap='plasma',edgecolor='none')
                ax.view_init(elev=90, azim=0)  # Set the top angle view
                ax.set_xlabel('Weight (w)')
                ax.set_ylabel('Bias (b)')
                ax.set_zlabel('Cost')
                ax.set_title('Cost Function Surface')
                ax.scatter(self.w, self.b, self.compute_cost_manual(self.w, self.b)+100, color='red', s=100)  # s is the size of the ball
                plt.pause(0.001)
            last_cost = self.compute_cost()
        plt.pause(3)
        print('w: {}, b: {}'.format(self.w,self.b))

    def plot_final_result(self):
        '''
        Plot the final result of linear regression
        '''
        plt.figure(figsize=(16, 9))
        plt.scatter(self.X, self.Y, color='blue', label='Data Points')
        plt.plot(self.X, self.predict(), color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Simple Linear Reçgression')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_learning_curve(self):
        plt.figure(figsize=(16, 9))
        plt.plot([i for i in range(len(self.cost_history))], self.cost_history, color='red')
        plt.xlabel('# of iterations')
        plt.ylabel('Cost')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.pause(3)
        plt.close()

    def plot_cost_function_3d(self):
        w_range = np.linspace(self.w - 40, self.w + 40, 100)
        b_range = np.linspace(self.b - 10, self.b + 10, 100)
        W, B = np.meshgrid(w_range, b_range)
        Cost = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Cost[i, j] = self.compute_cost_manual(W[i, j], B[i, j])

        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection='3d')
        ax.plot_surface(W, B, Cost, cmap='plasma',edgecolor='none')
        ax.set_xlabel('Weight (w)')
        ax.set_ylabel('Bias (b)')
        ax.set_zlabel('Cost')
        ax.set_title('Cost Function Surface')
        plt.show()

    def plot_cost_function_2d(self):
        '''
        Plot the function with variable w and constant b = 0
        '''
        fig = plt.figure(figsize=(16, 9))
        w_range = np.linspace(self.w - 2, self.w + 2, 100)
        cost_values = [self.compute_cost_manual(w, self.b) for w in w_range]
        plt.plot(w_range, cost_values, color='blue')
        plt.xlabel('Weight(w)')
        plt.ylabel('Cost')
        plt.title('Cost Function Surface')
        plt.show()


class MultipleLinearRegression(SimpleLinearRegression):
    def __init__(self, file):
        self.file = file
        values = np.array(pd.read_csv(file).values,'float')
        self.X = values[:,:-1]
        self.Y = values[:,-1]
        #Mean Normalization / Feature Scaling
        self.X = (self.X-np.mean(self.X))/(np.max(self.X)-np.min(self.X)) 
        self.w = 0
        self.b = 0
        self.cost_history = []


