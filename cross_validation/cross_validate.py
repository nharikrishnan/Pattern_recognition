"""class for creating cross validation data sets"""

import pandas as pd
import numpy as np

# class definition 
class cross_validation():
    """Cross validation to reduce overfitting
    """
    def __init__(self, matrix1, matrix2, split, seed = None):
        """
            initializing the required variables to perform cross_validation
            Args:
                matrix1: input values for the first class
                matrix2: input values for the second class
                seed: random initialization values, based on which the shuffling is performed
                split: this variable can take values random, same
        """
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.seed = seed
        self.split = split
        
    def create_folds(self, matrix1, matrix2, folds):
        """splits the data according to the number of folds specified as argumnet
            args:
                """
        N = (len(matrix1)+ len(matrix2))/folds
        mat_table1= []
        mat_table2= []
        while folds!=0:
            mat_table1.append(matrix1[0:int(N/2),])
            mat_table2.append(matrix2[0:int(N/2),])
            matrix1 = matrix1[int(N/2):,]
            matrix2 = matrix2[int(N/2):,]
            folds = folds-1
        return np.array(mat_table1), np.array(mat_table2)
    
    #cross_validation
    def cross_validate(self, mat_table1, mat_table2):
        """Function creates training and testing tables
            Args:
                mat_table1: cross validation training tables
                mat_table2: cross validation testing tables"""
        mat_table1 = np.array(mat_table1)
        mat_table2 = np.array(mat_table2)
        mat_train1 = []
        mat_train2 = []
        mat_test1 = []
        mat_test2 = []
        for i in range(len(mat_table1)):
            test_matrix1 = mat_table1[i]
            test_matrix2 = mat_table2[i]
            mat_test1.append(test_matrix1)
            mat_test2.append(test_matrix2)
            mat_table_temp1 = []
            mat_table_temp2 = []
            for j in range(len(mat_table1)):
                if j !=i:
                    mat_table_temp1.append(mat_table1[j])
                    mat_table_temp2.append(mat_table2[j])
            train_matrix1 = np.concatenate(mat_table_temp1)
            train_matrix2 = np.concatenate(mat_table_temp2)
            mat_train1.append(train_matrix1)
            mat_train2.append(train_matrix2)
        return mat_train1, mat_train2, mat_test1, mat_test2    #leave one out method
 
    def leave_one_out(self, matrix_train1, matrix_train2):
        matrix1 = matrix_train1
        matrix2 = matrix_train2
        train1 = []
        train2 = []
        test = []
        count1 =0
        count2 = 0
        for i in range(len(matrix1)):
            #print(count1)
            test_matrix = matrix1[i]
            train_matrix1 = np.delete(matrix1, i, axis = 0)
            train_matrix2 = matrix2
            test.append(test_matrix)
            train1.append(train_matrix1)
            train2.append(train_matrix2)
            count1 = count1+1
            ##### Function to train, predict and calculate accuracy
        for i in range(len(matrix2)):
            #print(count2)
            test_matrix = matrix2[i]
            train_matrix1 = matrix1
            train_matrix2 = np.delete(matrix2, i, axis = 0)
            test.append(test_matrix)
            train1.append(train_matrix1)
            train2.append(train_matrix2)
            count2 = count2 +1
    
        return train1, train2, test
