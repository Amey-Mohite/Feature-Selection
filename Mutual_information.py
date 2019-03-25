import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile

#classifyy
data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000) 
data.shape
numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['target'],axis=1),data['target'],test_size=0.3,random_state=0)
x_train.shape
x_test.shape


mi=mutual_info_classif(x_train.fillna(0),y_train)
mi
mi=pd.Series(mi)
mi.index=x_train.columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_= SelectKBest(mutual_info_classif,k=10).fit(x_train.fillna(0),y_train)
x_train.columns[sel_.get_support()]

#Regression
data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\houseprice_train.csv") 
data.shape
numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['SalePrice'],axis=1),data['SalePrice'],test_size=0.3,random_state=0)
x_train.shape
x_test.shape

mi=mutual_info_regression(x_train.fillna(0),y_train)
mi
mi=pd.Series(mi)
mi.index=x_train.columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_= SelectPercentile(mutual_info_regression,percentile=10).fit(x_train.fillna(0),y_train)
x_train.columns[sel_.get_support()]




