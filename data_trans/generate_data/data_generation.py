#class for generating random_vectors usig central limit theorem, and creating data point with the given mean and covariance
import numpy as np

class data_generation():
    
    def gausian_random(self, n, dimension):
        
        """ Function to generate gaussian random vectors with in the required dimension using central limit theorem
            Args:
                n = number of data points required
                dimension = the dimension of the data
         """
        mat = []
        for i in range(0,n):
            t = []
            for k in range(0,dimension):
                t.append(0)
            for l in range(0,12):
                for j in range(0,len(t)):
                    t[j] = t[j] + np.random.uniform()
            for m in range(0, len(t)):
                t[m] = t[m]-6
            mat.append(t)
        return np.array(mat)
    def generate_cov_data(self, matrix,covar, mean):
        
        """Function to generate  matrix with the given mean and covariance from gausian random vector
           Args:
               matrix = gaussian random vector with mean approximately 0 and covariance approximately 1  """
        val, vect = np.linalg.eig(covar.T)
        diag_val = np.diag(np.sqrt(val))
        x = np.matmul(vect, diag_val)
        w1 = np.matmul(x, matrix)
        w1_new = np.subtract(w1.T, np.array([np.mean(w1[0]), np.mean(w1[1]), np.mean(w1[2])]))
        w1_final = np.add(w1.T, mean)
        return w1_final
        