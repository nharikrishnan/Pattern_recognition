#importing required libraries
import numpy as np
import math

class bayesOptimal():
  
    def calculate_mean_vec(self,matrix1):
        """
        returns the mean vector for the matrix
        args:
            matrix: matrix for which the mean vector needs to be calculated
        """
        mean_vec = []
        matrix_transposed = matrix1.T
        for t in matrix_transposed:
            mean_vec.append(np.mean(t))
        return mean_vec

    def fit_model(self, cov_matrix1, cov_matrix2, mean1, mean2):
        """
        Retuns the Coefficients comparing the posterior probability
        Args:
            matrix1: matrix for class one
            matrix2: matrix for class two
        """
        #calculating A
        #X = combine_classes
#        cov_matrix1 = np.cov(matrix1.T)
#        cov_matrix2 = np.cov(matrix2.T)
        cov_inv_matrix1 = np.linalg.inv(cov_matrix1)
        cov_inv_matrix2 = np.linalg.inv(cov_matrix2)
        A = (np.subtract(cov_inv_matrix2, cov_inv_matrix1))/2

        #calculating B
        #mean
#        mean1 = np.array(self.calculate_mean_vec(matrix1))
#        mean2 = np.array(self.calculate_mean_vec(matrix2))
        B = np.matmul(mean1.T,cov_inv_matrix1) - np.matmul(mean2.T,cov_inv_matrix2)

        #Claculating C

        det_cov1 = np.linalg.det(cov_matrix1)
        det_cov2 = np.linalg.det(cov_matrix2)
        c1 = math.log(det_cov2/det_cov1)
        c2 = np.matmul(np.matmul(mean2.T,cov_inv_matrix2), mean2)
        c3 = np.matmul(np.matmul(mean1.T,cov_inv_matrix1), mean1)
        C = (c1+c2-c3)/2

        return A, B, C

    def predict(self, matrix_test, A, B, C):
        """returns the predicted """
        classifier_result = []
        if len(np.array(matrix_test).shape) ==1:
            matrix_test = matrix_test.tolist()
            term1 = np.matmul(np.matmul(np.transpose(matrix_test), A), matrix_test)
            term2 = np.matmul(B, matrix_test)
            if term1 + term2 + C >0:
                matrix_test.append('0')
            else:
                matrix_test.append('1')
            classifier_result.append(matrix_test)

        else:

            for t in matrix_test.tolist():
                term1 = np.matmul(np.matmul(np.transpose(t), A), t)
                term2 = np.matmul(B, t)
                if term1 + term2 + C >0:
                    t.append('0')
                else:
                    t.append('1')
                classifier_result.append(t)

        return np.array(classifier_result)    #def accuracy:
