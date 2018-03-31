import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


# checking feature balance
raw_data = pd.read_csv('data.csv')
for i in raw_data.columns:
    x = raw_data[i]
    plt.bar([0,1], [raw_data[i].count() - raw_data[i].astype(bool).sum(axis=0),raw_data[i].astype(bool).sum(axis=0)],width=0.1)
    print(i)
    plt.show()

# droping columns where most of the values are zeros
raw_data.drop(['E3', 'E11', 'V5'], axis=1)

# splitting into training and testing
X = raw_data.iloc[:,1::]
Y = raw_data.iloc[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)

# feature scaling/Normalization of training and testing data

col_list = ['P1', 'P2', 'P3', 'P4', 'P6', 'P7', 
            'E1', 'E2', 'E3', 'E4','E6', 'E7', 'E8', 'E10', 'E11',
            'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V9']

for i in col_list:
    Xtrain[i] = (Xtrain[i] - Xtrain[i].mean())/(Xtrain[i].max() - Xtrain[i].min())

for i in col_list:
    Xtest[i] = (Xtest[i] - Xtest[i].mean())/(Xtest[i].max() - Xtest[i].min())

# Logistic Regression
lr = LogisticRegression(C=0.5)
lr.fit(Xtrain,ytrain.values.ravel())

# Training Accuracy
lr.score(Xtrain, ytrain)

# Testing Accuracy
ypred = lr.predict(Xtest)
print(accuracy_score(ytest, ypred))
print(classification_report(ytest, ypred))

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(Xtrain,ytrain.values.ravel())

# Training Accuracy
bnb.score(Xtrain, ytrain)

# Testing Accuracy
ypred = bnb.predict(Xtest)
print(accuracy_score(ytest, ypred))
print(classification_report(ytest, ypred))

# Random Forests
rfc = RandomForestClassifier()
rfc.fit(Xtrain,ytrain.values.ravel())

# Training Accuracy
rfc.score(Xtrain, ytrain)

# Testing Accurcy
ypred = rfc.predict(Xtest)
print(accuracy_score(ytest, ypred))
print(classification_report(ytest, ypred))
