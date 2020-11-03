#Using logistic regression to predict on what kinds of employees are prone to leave company X.
#I chose logistic regression classifier because the dependent variable has two dichotomous values

#Data preprocessing

#Import libraries
import numpy as np
import pandas as pd
from sklearn import metrics

#importing the dataset
dataset = pd.read_csv('data.csv')

#Select the independent and dependent variables
X = dataset.iloc[:, 1:10].values
#print(X)
Y = dataset.iloc[:,-1].values

#no missing values

#encode categorical data
print(dataset.info())  #'dept' and 'salary' are categorical data
from sklearn.preprocessing import LabelEncoder
categories = LabelEncoder()

X[:,-2] = categories.fit_transform(X[:,-2])
#print(X[:,-2]) #encodes the dept values: 7 for sales, 4 for management, 5 for marketing, 0 for IT, etc.
X[:,-1] = categories.fit_transform(X[:,-1])
#print(X[:,-1])  #encodes the salary values: 1 for low, 2 for medium and 0 for high

#split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y, test_size = 0.25,random_state=0
)

#feature scaling
#print(dataset.describe()) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#........................................................
#Implementing the logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(X_train,Y_train)


#predict test results
Y_pred= classifier.predict(X_test)

#print('Report: {}'.format(metrics.classification_report(Y_test, Y_pred)))

#compare results and look at metrics
print('Accuracy: {}%'.format(round(metrics.accuracy_score(Y_test, Y_pred)*100,1)))
print('Precision: {}%'.format(round(metrics.precision_score(Y_test, Y_pred)*100,1)))
print('Recall: {}%'.format(round(metrics.recall_score(Y_test, Y_pred)*100,1)))





























