#Fishers LDA
import pandas as pd
import numpy as np

class FishersLDA():
    
    def __init__(self, matrix1, matrix2):
        """initializing the input data for which the fishers discriminant needs to be computed"""
        self.matrix1 = matrix1
        self.matrix2 = matrix2
    
    def compute_mean(self, X):
        """Function to compute the mean"""
        return np.mean(X, axis = 1)

#     def scatter_matrix(self, matrix):
#         """function to compute the scatter"""
#         mean_matrix = np.mean(matrix, axis = 0)
#         len_matrix = matrix.shape[0]
#         matrix = matrix - mean_matrix
#         covariance = matrix.T.dot(matrix)
#         return covariance

    def scatter_matrix(self, matrix):
        """function to compute the scatter"""
        covariance = np.cov(matrix)
        return covariance

    
    def fld(self):
        """Function to calculate the W, the direction in which the datapoints needs to be projected"""
        m0 = self.compute_mean(self.matrix1.T)
        m1 = self.compute_mean(self.matrix2.T)
        term1 = m0 - m1
        s1 = self.scatter_matrix(self.matrix1.T)
        s2 = self.scatter_matrix(self.matrix2.T)
        sw = s1+s2
        sw_inv = np.linalg.inv(sw)
        W = np.matmul(sw_inv, term1)
        return W
    
    def compute_density(self, X, mean, var):
        """function to find the gausian density
            Args:
                X: features
                xi: center"""
        a=(np.power((X-mean),2))/(2*(var**2))
        term1 = 1/(var*np.sqrt(np.pi * 2))
        term2 = np.exp(-a)
        density = term1*term2
        return density

    def gaussian_parameters(self):
        """function to estimate the parameters for the projected data for both the classes, assuming is in a gaussian distribution
        """
        w = self.fld()
        X = np.append(self.matrix1, self.matrix2, axis = 0)
        X_projected = X.dot(w.T)
        matrix1_proj = X_projected[:200]
        matrix2_proj = X_projected[-200:]
        n = len(X_projected)
        prior1 = len(matrix1_proj)/n
        prior2 = len(matrix2_proj)/n
        mean_c1 = np.mean(matrix1_proj)
        mean_c2 = np.mean(matrix2_proj)
        var_c1 = np.cov(matrix1_proj)
        var_c2 = np.cov(matrix2_proj)
        #var_c2 = self.scatter_matrix(matrix2_proj)/(len(matrix2_proj)-1)
        return mean_c1, mean_c2, var_c1, var_c2, prior1, prior2

    def predict(self, test):
        """Predicting the class values for the testing data points
            Args:
                 test: testing data points"""
        # projecting the data in the direction W
        w = self.fld()
        test_projected = test.dot(w)
        mean_c1, mean_c2, var_c1, var_c2, prior1, prior2 = self.gaussian_parameters()
        predictions = []
        c1_den = []
        c2_den = []
        for i in range(len(test_projected)):
            c1_density = self.compute_density(test_projected[i], mean_c1, var_c1)
            c2_density = self.compute_density(test_projected[i], mean_c2, var_c2)
            c1_den.append(c1_density)
            c2_den.append(c2_density)
            if c1_density > c2_density:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions