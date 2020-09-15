####bayesian estimate

import pandas as pd
import numpy as np

####class definition####
class bayesian_estimate():
    """Functions for estimating the mean using bayes estimation method"""
    def __init__(self, matrix, abs_mean, abs_covariance, n = None):
        """intializing the required varaibles
            Args:
                 matrix1: The data points for which estimation needs to be done
                 abs_mean: The assumed mean of the data points
                 sigmaL The assumed covariace of data points"""
                 
        self.matrix = matrix
        self.abs_mean = abs_mean
        self.abs_covariance = abs_covariance
        self.n = n
        if self.n:
            self.matrix = matrix[:self.n]
        else:
            self.matrix = matrix

    def mean_estimate(self):
        """functon which returns the mean estimate for the data points passes """
        shape = self.matrix.shape[1]
        sigma_0 = np.identity(shape)
        sigma = self.abs_covariance
        mean_0 = self.abs_mean
        mean = np.mean(self.matrix.T, axis = 1)
        n = len(self.matrix)
        inv_sigma = np.linalg.inv(np.add(sigma_0,(1/n)*(sigma)))
        term1 = np.matmul(sigma_0, inv_sigma)
        term2 = np.matmul(sigma, inv_sigma)
        mean_estimate = np.matmul(term1, mean.T) + (1/n* (np.matmul(term2, mean_0)))
        return mean_estimate