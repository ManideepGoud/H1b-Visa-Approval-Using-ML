# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:51:25 2020

@author: Manideep Goud
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv('h1b_kaggle.csv',nrows=30000)

Dataset.drop(["Sno"],axis=1,inplace=True)
Dataset.drop(["YEAR"],axis=1,inplace=True)
Dataset.drop(["lon"],axis=1,inplace=True)
Dataset.drop(["lat"],axis=1,inplace=True)

Soc_Name=pd.get_dummies(Dataset["SOC_NAME"],drop_first=True)
Dataset=pd.concat([Soc_Name,Dataset],axis=1)
Dataset.drop(["SOC_NAME"],axis=1,inplace=True)
Job=pd.get_dummies(Dataset["JOB_TITLE"],drop_first=True)
Dataset=pd.concat([Job,Dataset],axis=1)
Dataset.drop(["JOB_TITLE"],axis=1,inplace=True)
FT=pd.get_dummies(Dataset["FULL_TIME_POSITION"],drop_first=True)
Dataset=pd.concat([FT,Dataset],axis=1)
Dataset.drop(["FULL_TIME_POSITION"],axis=1,inplace=True)

WS=pd.get_dummies(Dataset["WORKSITE"],drop_first=True)
Dataset=pd.concat([WS,Dataset],axis=1)
Dataset.drop(["WORKSITE"],axis=1,inplace=True)

Employer_Name=pd.get_dummies(Dataset["EMPLOYER_NAME"],drop_first=True)
Dataset=pd.concat([Employer_Name,Dataset],axis=1)
Dataset.drop(["EMPLOYER_NAME"],axis=1,inplace=True)

X=Dataset.drop(['CASE_STATUS'],axis=1)
y=Dataset['CASE_STATUS']

Status=pd.get_dummies(Dataset["CASE_STATUS"])
y=pd.concat([Status,y],axis=1)
y.drop(["CASE_STATUS"],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

from sklearn.tree import DecisionTreeClassifier
DT= DecisionTreeClassifier()
DT.fit(X_train, y_train)

cls_pred=DT.predict(X_test)

from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_test,cls_pred)
print(acc1)


from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 6)
RFclassifier.fit(X_train, y_train)
cls_pred1=RFclassifier.predict(X_test)
from sklearn.metrics import accuracy_score
acc2=accuracy_score(y_test,cls_pred1)
print(acc2)