"""This module contains all the required functions for creating a perceptron for classfication"""

#import statements
import numpy as np

#class definition
class perceptron():
    
    def __init__(self, input_dim, iteration = 100, rate = 0.0001):
        self.input_dim = input_dim
        self.iteration = iteration
        self.rate = rate
        self.weights = np.ones(input_dim + 1)
    
    def combine_classes(self, train1, train2):
        train = np.append(train1, train2, axis = 0)
        n1 = len(train1)
        n2 = len(train2)
        output1 = [0]*n1
        output2 = [1]*n2
        output = np.append(np.array(output1), np.array(output2), axis = 0)
        return train, output
                           
    def predict(self, X):
        result = []
        sum1 = np.dot(X, self.weights[1:]) + self.weights[0]
        for i in sum1:
            if i >0:
                result.append(1)
            else:
                result.append(0)
        return result
    
    def prediction(self, inputs):
        summation = np.dot(inputs.T, self.weights[1:]) + self.weights[0]
        if summation >= 0:
            activation = 1
        else:
            activation = 0            
        return activation
        
    def train(self, X, Y):
        for i in range(self.iteration):
            for x, y in zip(X, Y):
                prediction = self.prediction(x)
                self.weights[1:] = self.weights[1:] + (self.rate * (y - prediction) * x)
                self.weights[0] = self.weights[0] +  (self.rate * (y - prediction))
        return self.weights