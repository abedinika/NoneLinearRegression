"""
Designed and developed by: Nika Abedi - contact email: nikka.abedi@gmail.com
*****************************************************************************
None Linear Regression
*****************************************************************************

"""
import numpy as np
import matplotlib.pyplot as plt


class NLR:
    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def __init__(self):
        # the points on the x axis for plotting
        self.x = np.linspace(0,10,100)
        self.x = np.expand_dims(self.x, 1)
        # compute the value (amplitude) of the sin wave at the for each sample
        self.y = np.sin(self.x)
        self.y = np.expand_dims(self.y, 1)

        # define objective function variables -> a1x^2 + a2x + c
        self.a1 = np.random.rand(1)
        self.a = np.random.rand(1)
        self.c = np.random.rand(1)

        self.epoch = 10
        self.alpha = [0.1, 0.2]

    # -------------------------------------------------------------
    # Plot the out puts
    # -------------------------------------------------------------
    def visulize(self):
        # plot the main function
        plt.scatter(self.x, self.y)

    # -------------------------------------------------------------
    # Gradient descent based on a1
    # Gradient based on a1 => -2yx^2 + 2a1x^4 + 2ax^3
    # -------------------------------------------------------------
    def grad_a1(self, a1):
        d_a1 = -2 * np.matmul(np.power(self.x, 2), self.y) + 2 * a1 * np.power(self.x, 4) +\
               2 * self.a * np.power(self.x, 3)

        return d_a1

    # -------------------------------------------------------------
    # Gradient descent based on a
    # Gradient based on a1 => -2yx + 2a1x^3 + 2ax^2
    # -------------------------------------------------------------
    def grad_a(self, a):
        d_a = -2 * np.matmul(self.x, self.y) + 2 * self.a1 * np.power(self.x, 3) +\
              2 * a * np.power(self.x, 2)

        return d_a

    # -------------------------------------------------------------
    # Objective function
    # -------------------------------------------------------------
    def objfunc(self):
        obj = np.linalg.norm((self.y - (self.a1 * self.x**2) + (self.a * self.x)))**2
        return obj

    # -------------------------------------------------------------
    # Minimization Optimum based on a1 and a
    # -------------------------------------------------------------
    def minimization(self):
        for iter in range(self.epoch):
            for counter in range(self.x.size):
                self.a1 = self.a1 - (self.alpha[0] * self.grad_a1(self.x[counter]))
                self.a = self.a - (self.alpha[0] * self.grad_a(self.x[counter]))
            print('\nIteration number:\n', iter)
            print('\nObjective function:\n', self.objfunc())

    # -------------------------------------------------------------
    # Error estimation - MSE
    # -------------------------------------------------------------
    def err(self):
        mse = np.array([])
        for counter in range(self.x.size):
            mse[counter] = np.power((self.y[counter] - np.power((self.a1, self.x[counter]), 2) +
                                     self.a * self.x[counter]), 2)
        return mse