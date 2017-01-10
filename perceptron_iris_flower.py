from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class YourNameMLP():
    def __init__(self, eta=0.02):
        self.eta = eta
        self.epochs = num_epochs

    def propagate(self, X):
        return np.where(self.activation(X) < 1.0, 0, np.where(self.activation(X) < 2.0, 1, 2))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def first_line(self):
        return 1

    def second_line(self):
        return 2

    def plot(self, dimension, dimension_name):
        x0 = []
        x1 = []
        x2 = []

        xp0 = []
        xp1 = []
        xp2 = []

        i = 0
        yFull = mlp.activation(X)

        y0 = []
        y1 = []
        y2 = []

        yp0 = []
        yp1 = []
        yp2 = []

        dim = dimension

        for line in X:
            #Original
            if y[i] == 0:
                x0.append(line[dim])
                y0.append(yFull[i])
            elif y[i] == 1:
                x1.append(line[dim])
                y1.append(yFull[i])
            else:
                x2.append(line[dim])
                y2.append(yFull[i])

            #Predict
            if predict[i] == 0:
                xp0.append(line[dim])
                yp0.append(yFull[i])
            elif predict[i] == 1:
                xp1.append(line[dim])
                yp1.append(yFull[i])
            else:
                xp2.append(line[dim])
                yp2.append(yFull[i])

            i += 1

        maxX0 = np.amax(x0)
        maxX1 = np.amax(x1)
        maxX2 = np.amax(x2)

        maxX = np.maximum(maxX0, np.maximum(maxX1, maxX2))

        ######### Original #########
        plt.figure('Original')
        plt.subplot(221+dim)
        plt.plot(x0, y0, 'ro')
        plt.plot(x1, y1, 'bo')
        plt.plot(x2, y2, 'go')

        plt.plot([0,maxX], [mlp.first_line(), mlp.first_line()])
        plt.plot([0,maxX], [mlp.second_line(), mlp.second_line()])

        plt.xlabel(dimension_name)
        plt.ylabel('activation')

        ######### Predicao #########
        plt.figure('Predict')
        plt.subplot(221+dim)
        plt.plot(xp0, yp0, 'ro')
        plt.plot(xp1, yp1, 'bo')
        plt.plot(xp2, yp2, 'go')

        plt.plot([0,maxX], [mlp.first_line(), mlp.first_line()])
        plt.plot([0,maxX], [mlp.second_line(), mlp.second_line()])

        plt.xlabel(dimension_name)
        plt.ylabel('activation')
        return self

    def plot_cost(self):
        plt.figure('Cost')
        t = np.arange(0, self.epochs, 1)
        plt.plot(t, self.cost_, lw = 2)
        plt.grid(True)

        plt.xlabel('epoch')
        plt.ylabel('cost')
        return self

    def learn(self, X, y, reinitialize_weights=True):
        if reinitialize_weights:
            self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = (target - output)
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error

            cost = ((y - self.activation(X)) ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self


if __name__ == '__main__':
    iris_data = load_iris()
    print(iris_data.keys())
    print(iris_data['DESCR'])
    n_samples, n_features = iris_data.data.shape
    print('Numero de amostras de entrada:', n_samples)
    print('Numero de atributo em cada amostra de entrada:', n_features)
    print('A primeira amostra:', iris_data.data[0])
    print('Dimensoes das entradas:', iris_data.data.shape)
    print('Dimensoes das classes:', iris_data.target.shape)
    print(iris_data.target)
    X = iris_data.data  # you may need to transform the data
    y = iris_data.target
    num_epochs = 100  # you can change this at will
    mlp = YourNameMLP()

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        mlp.learn(X, y)

    print('Pesos:', mlp.w_)
    print('As predicoes sao:', mlp.propagate(X))

    predict = mlp.propagate(X)
    mlp.plot(0, 'sepal length')
    mlp.plot(1, 'sepal width')
    mlp.plot(2, 'petal length')
    mlp.plot(3, 'petal width')

    mlp.plot_cost()

    plt.show()


