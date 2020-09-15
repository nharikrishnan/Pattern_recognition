#import statements
import numpy as np
import pandas as pd

# ML estimate
class mlEstimate():
    def __init__(self,matrix, n = None):
        self.n = n
        if self.n:
            self.matrix = matrix[:self.n]
        else:
            self.matrix = matrix
    def mean_estimate(self):
        test = self.matrix.T
        mean_vec = []
        for i in range(test.shape[0]):
            mean_vec.append(np.mean(test[i]))
        return mean_vec
    def cov_estimate(self):
        mean_matrix = np.mean(self.matrix, axis = 0)
        len_matrix = self.matrix.shape[0]
        self.matrix = self.matrix - mean_matrix
        covariance = self.matrix.T.dot(self.matrix)/(len_matrix-1)
        return covariance
