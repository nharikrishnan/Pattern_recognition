"""KNN classification"""

#import statements
import numpy as np
import pandas as pd
import math

#class definition

class KNN():
    """K nearest neighbour class"""
    def __init__(self,matrix_class1, matrix_class2, k=3):
        """ Initializing the class object
            Args:
                matrix_class1: training data for the first class
                matrix_class2: training data for the second class
                test: testing data points
                k: Number of nearest neighbours to be considered"""
       
        self.matrix_class1 = matrix_class1
        self.matrix_class2 = matrix_class2
        self.k = k
    
    def euclidean_dist(self, vec1, vec2):
        """ Function to calculate eucleadean distance between 2 vectors 
            returns the eucleadian distance between the vectors passed as argumnets
            Args:
                vec1: first vector
                vec2: second vector"""
        distance = 0
        for i in range(len(vec1)):
            distance = distance + (vec1[i] - vec2[i])**2
        return distance
    
    def knn_classification(self, test):
        """function to classify the testing datapoint:
            returns the class value for binary testing data points"""
        result = []
        for i in test:
            data_point = i
            distances = []
            for j in self.matrix_class1:
                distances.append(self.euclidean_dist(j, data_point))
            for j in self.matrix_class2:
                distances.append(self.euclidean_dist(j, data_point))
            matrix1 = []
            matrix2 = []
            for i in self.matrix_class1.tolist():
                i.append(0)
                matrix1.append(i)
            for i in self.matrix_class2.tolist():
                i.append(1)
                matrix2.append(i)
            matrix = list(matrix1) + list(matrix2)
            final_list = []
            for i in range(len(matrix)):
                temp = matrix[i]
                temp.append(distances[i])
                final_list.append(temp)
            length = np.array(final_list).shape[1] -1
            matrix_sorted = sorted(matrix, key=lambda a_entry: a_entry[length]) 
            count  = 0
            for i in matrix_sorted[:self.k]:

                if i[length-1] == 0:
                    count+=1
            if count >= math.ceil(self.k/2):
                result.append(0)
            else:
                result.append(1)
        return result