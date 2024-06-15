from module.linear_regression import SimpleLinearRegression
from module.linear_regression import MultipleLinearRegression
from module.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

if __name__=='__main__':
    # lr = SimpleLinearRegression('../include/dataset/linear_regression.csv')
    # lr.fit(0.5, plot=True)
    # print("MSE: ", lr.compute_cost())
    # lr.plot_cost_function_3d()

    # lr = MultipleLinearRegression('../include/dataset/multiple_LR.csv')
    # lr.fit(1, plot=True)
    # print("MSE: ", lr.compute_cost())

    df = datasets.load_breast_cancer()
    X, Y = df.data, df.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    classifier = LogisticRegression(X_train, Y_train)
    classifier.fit(0.5)
    print(classifier.accuracy(classifier.predict(X_train), Y_train))

