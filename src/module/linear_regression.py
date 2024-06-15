import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
import plotly.graph_objs as go

class SimpleLinearRegression():
    def __init__(self, file):
        self.file = file
        values = np.array(pd.read_csv(file).values,'float')
        self.X = values[:,0]
        self.Y = values[:,1]
        #Mean Normalization / Feature Scaling
        self.X = (self.X-np.mean(self.X))/(np.max(self.X)-np.min(self.X)) 
        self.w = np.zeros(len(self.X[0]))
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
    def __init__(self, X_train, y_train):
        # self.file = file
        # values = np.array(pd.read_csv(file).values,'float')
        self.X, self.Y = X_train, y_train
        #Feature Scaling is more essential now to make the gradient descend faster
        self.mean, self.std = np.mean(self.X, axis=0), np.std(self.X, axis=0)
        self.X = self._zscore_normalize(self.X)
        self.w,self.b = None, None
        self.cost_history = []

    def _zscore_normalize(self, X):
        return (X-self.mean) / self.std
    
    def predict(self, X):
        X = self._zscore_normalize(X)
        return np.dot(X, self.w) + self.b
    
    def compute_cost(self):
        m,_ = self.X.shape
        y_hat = np.dot(self.X, self.w) + self.b
        return 0.5/m  * np.sum((y_hat - self.Y)**2)

    def fit(self, alpha, tolerance=1e-8, plot=None):
        #last_cost = np.inf
        m,_ = self.X.shape
        epoch = 0
        self.w = np.ones(len(self.X[0]))
        self.b = 0
        while epoch<1000:
            print('Epoch: {}, w: {}, b: {}, cost: {}'.format(epoch, self.w, self.b, self.compute_cost()))
            y_hat = np.dot(self.X, self.w) + self.b
            dJ_dw = np.dot(self.X.T, (y_hat-self.Y))/m
            dJ_db = np.sum(y_hat-self.Y)/m
            self.w = self.w - (alpha * dJ_dw)
            self.b = self.b - (alpha * dJ_db)
            self.cost_history.append(self.compute_cost())
            #last_cost = self.compute_cost()
            epoch += 1
        print('Epoch: {}, w: {}, b: {}, cost: {}'.format(epoch, self.w, self.b, self.compute_cost()))


    def plot_learning_curve(self):
        plt.figure(figsize=(16, 9))
        plt.plot([i for i in range(len(self.cost_history))], self.cost_history, color='red')
        plt.xlabel('# of iterations')
        plt.ylabel('Cost')
        plt.title('Learning Curve')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, X_test, y_test):
        predictions = self.predict(X_test)
        plt.figure(figsize=(16, 9))
        plt.scatter(y_test, predictions, color='blue')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        plt.show()

    def plot_3d_regression(self):
        x1, x2 = self.X[:, 0], self.X[:, 1]
        y = self.Y

        trace1 = go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode='markers',
            marker=dict(
                size=5,
                color='blue',                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )
        )

        plane_x = np.linspace(min(x1), max(x1), 10)
        plane_y = np.linspace(min(x2), max(x2), 10)
        plane_x, plane_y = np.meshgrid(plane_x, plane_y)
        plane_z = self.w[0] * plane_x + self.w[1] * plane_y + self.b

        trace2 = go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            opacity=0.5,
            showscale=False
        )

        data = [trace1, trace2]
        layout = go.Layout(
            title='3D Linear Regression',
            scene=dict(
                xaxis=dict(title='Feature 1'),
                yaxis=dict(title='Feature 2'),
                zaxis=dict(title='Target')
            )
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2
    
    X, y = datasets.make_regression(n_samples=10000,n_features=2,noise=20,random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    
    regressor = MultipleLinearRegression(X_train, y_train)
    regressor.fit(0.01)
    prediction = regressor.predict(X_test)
    regressor.plot_3d_regression()
    print("Accuracy: ", r2_score(y_test, prediction))
    print("MSE: ", regressor.compute_cost())
        


