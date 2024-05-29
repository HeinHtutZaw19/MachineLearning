from module.linear_regression import SimpleLinearRegression

if __name__=='__main__':
    lr = SimpleLinearRegression('../include/dataset/linear_regression.csv')
    lr.fit(0.5, plot=True)
    print("MSE: ", lr.compute_cost())
    lr.plot_cost_function_3d()
