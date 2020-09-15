"""implemention of parzen window non parametric estimates"""
#Parzen window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

class ParzenWindow():
    """Non parametrix parzen window estimiate, the window function used is gaussian
        Assumes that the feature is one dimensional"""
    def __init__(self,X,h):
        """variable initialization
            Args:
                X: features
                h: width of gaussian"""
        self.X = X
        self.h = h

    def window(self, X, xi, h):
        """function to create the window
            Args:
                X: features
                xi: center"""
        a=(np.power((X-xi),2))/(2*(h**2))
        term1 = 1/(h*np.sqrt(np.pi * 2))
        term2 = np.exp(-a)
        density = term1*term2
        return sum(density)/200

    def fn_density(self):
        """returns the normalized density estimates for the set of points passed"""
        fx_dens = []
        for i in self.X:
            fx_dens.append(self.window(self.X, i, self.h))
        fx_norm = fx_dens/sum(fx_dens)
        return(fx_norm)
    
    def expectation(self,fx_norm):
        """returns the estimated mean using the parzen window
            Args
                fx_norm: normalized density estimates"""
        sum1 = 0
        for i in range(len(self.X)):
            sum1 = sum1 + (self.X[i]*fx_norm[i])
        return sum1
    
    def variance(self, fx_norm):
        """returns the estimated mean 
            Args:
                fx_norm: normalized density estimates
        """
        
        cov_x = 0
        for i in range(len(self.X)):
            cov_x = cov_x + np.power((self.X[i] - self.expectation(fx_norm)), 2)
        return cov_x/(len(self.X)-1)
    
    def plot_estimates(self, fx_norm, X_label, Y_label):
        """Plots the estimated densities
            Args:
                fx_norm: normalized density estimates
                X_label: label for the X axis
                Y_label: label for the Y axis"""
        #plt.plot(X, fx_norm)
        plot.style = 'seaborn-notebook'
        fig = plot.figure()
        ax = fig.add_subplot(111)
        plt.scatter(self.X,density_approx_x1)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, .01)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plot.xlabel(X_label)
        plot.ylabel(Y_label)