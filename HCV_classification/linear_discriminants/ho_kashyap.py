""" Date:
    Owner:
This module contains all the required functions for ho-kashyp method"""

#import statements
import pandas as pd
import numpy as np

#class definition

class HK(object):
    def _init_(self):
        self.W=None
    
    def combine_classes(self, train1, train2):
        train = np.append(train1, train2, axis = 0)
        n1 = len(train1)
        n2 = len(train2)
        output1 = [-1]*n1
        output2 = [1]*n2
        output = np.append(np.array(output1), np.array(output2), axis = 0)
        return train, output

        
    def train(self,x,y,lr=0.0001,num_iters=100):
        num_train,num_features=x.shape
        xone=np.ones((num_train,1))
        x=np.column_stack((xone,x))
        x[np.where(y==-1)]=x[np.where(y==-1)]*-1
        self.W=np.ones((num_features+1))
        Y=x
        Ywn=np.linalg.pinv(Y)
        bias=np.ones((num_train,))
        bmin=0
        flag=False
        for i in range (num_iters):
            temp=np.array(np.dot(Y,self.W))
            Evector=temp-bias
            Evector_=1/2*np.add(Evector,np.absolute(Evector))
            bias=np.add(2*lr*Evector_,bias)
            self.W=np.dot(Ywn,bias)
            for row in Evector_:
                if (np.absolute(row)<=bmin):
                    print('reach bmin,end')
                    flag=True
                    break
            if flag:
                break
        return self.W       
         
            
    def linear(self,x):
        return np.dot(x,self.W[1:])+self.W[0]
    
    def predict(self,x):
        return np.where(self.linear(x)>=0.0,1,-1)