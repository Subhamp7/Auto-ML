#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:43:02 2021

@author: subham
"""

#loading the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from flaml import AutoML


#loading the dataset
data=pd.read_csv("bank.csv")

#checking the count of missing data
print("The count of missing data is :",data.isnull().sum().sum())

def corr_max(dataset):
  dataset_corr = dataset.corr().abs()
  ones_matrix = np.triu(np.ones(dataset_corr.shape), k=1).astype(np.bool)
  dataset_corr = dataset_corr.where(ones_matrix)
  column_drop = [index for index in dataset_corr.columns if any(dataset_corr[index] > 0.80)]
  dataset=dataset.drop(column_drop, axis=1)
  return dataset

#splitting the dependent and independent variables
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

#scaling
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting data into test and train data
x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

automl = AutoML()
automl_settings = {
    "time_budget": 30,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "bank_model.log",
}

#automl.fit(x_train, y_train, task="classification")
automl.fit(x_train, y_train, **automl_settings)


model_details  =automl.__dict__
pred=automl.predict(x_test)
cm=confusion_matrix(pred,y_test)

# Predict
pred_1 = automl.predict(x_test)
# Export the best model
ml_model = automl.model
