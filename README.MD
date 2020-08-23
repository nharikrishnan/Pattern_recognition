
# Performance of classification algorithms on HCV- Egypt dataset

### Overview
Hepatitis C virus (HCV) is a wide-spread disease in many countries and unlike the A and B variations of this virus, there is no vaccine for the HCV[1]. The HCV is an inflammation of the liver caused due to an RNA virus and it affects more than 170 million people worldwide. A major difficulty in identifying the HCV infection is that it’s asymptomatic with very little visible symptoms during the primary stages[2]. Liver biopsy is still the most widely used technique to diagnose the staging of liver fibrosis(HCV), this method is complicated, painful and very expensive. If failed to diagnose, HCV can be fatal[2]. Diagnosing the disease and identifying the correct stage of the disease and providing effective treatment based on the severity of the disease is the right way to ensure quality service for patients. Intervention during the early stages of HCV can even prevent the development of liver cancer. Previous studies show that machine learning and classification techniques have a very good probability of predicting the correct stages of HCV by independently analyzing the features from the data[3][4]. Identifying the correct stage and treating based on the level of infection is the only possible way. This project aims to explore and compare the performances of different classification techniques in classifying the acute and chronic cases of the hepatitis C virus.

 ### Methods
  <ul>
    <li>quadratic classifiers</li>
    <ul><li> bayesian Estimate </li>
        <li>maximum likelihood estimates</li>
    </ul>
    <li>Nearest Neighbour( K-NN)</li>
    <li>Linear Discriminants</li>
    <ul><li>Fishers LDA </li>
        <li>Perceptron</li>
        <li>Hokashyap</li>
    </ul>
  </ul>

 The study also analyses the effect of Simultaneous diagonalization on the classes.The top 3 accuracies were for quadratic classifier with bayesian estimation(60%), followed by linear classifiers fisher’s LD(56%) and Ho-Kashyap(55%). All the testings were done using 5 fold cross-validation. After diagonalizing the data points the accuracy was nearly the same, there wasn’t much variation
## Getting Started

Development was done using python version 3.7.7 and only numPy package was used to code the classification algorithms, Pandas was used to process the data. 

### Prerequisites

### Prerequisites
Python version 3.7.7 was used for development.<br>
Python Packages required can be found in <i>'requirement.txt'</i><br>
The packages can be installed using the command:
```
pip install 'package_name'
```
Pretrained glove.6b.50d.txt word embedding was used in the deep learning models to assign weights, the word embedding file can be downloaded from https://nlp.stanford.edu/projects/glove/. The downloaded file must be placed in the word embedding folder, any other pretrained word embedding can be also be used.
