import numpy as np

class feature_operations():
    
    def extract_eigen(self,matrix):
        #calculating the eigenvalue_t and sqrt inv of eigen value of covariance
        val, vec = np.linalg.eig(np.cov(matrix.T))
        vec_t = np.transpose(vec)
        daig_val = np.diag(1. / np.sqrt(val))
        return daig_val, vec_t

    def whiten_data(self,matrix):
        daig_val, vec_t = self.extract_eigen(matrix)
        return np.matmul(daig_val,np.matmul(vec_t, matrix.T))
    
    def simultaneous_diag(self,mat1, mat2, transformation ='v'):
        mat1_diag_val, mat1_vec_t = self.extract_eigen(mat1)
        #first transformation
        mat1_y = np.matmul(mat1_vec_t, mat1.T)
        mat2_y1 = np.matmul(mat1_vec_t, mat2.T)
        #second transformation
        mat1_z = np.matmul(mat1_diag_val,mat1_y)
        mat2_z1 = np.matmul(mat1_diag_val,mat2_y1)
        #3rd transformation
        mat2_z1_diag, mat2_z1_vec_t = self.extract_eigen(mat2_z1.T)
        mat1_v = np.matmul(mat2_z1_vec_t, mat1_z)
        mat2_v = np.matmul(mat2_z1_vec_t, mat2_z1)
        if transformation.lower() == 'v':
            return mat1_v, mat2_v
        if transformation.lower() == 'z':
            return mat1_z, mat2_z1
        if transformation.lower() == 'y':
            return mat1_y, mat2_y1