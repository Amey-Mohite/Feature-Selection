import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import f_classif,f_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile

#classification
data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000)  
data.shape
data.head()

numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['target','ID'],axis=1),data['target'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape


univariate = f_classif(x_train.fillna(0),y_train)
univariate

univariate = pd.Series(univariate[1])
univariate.index=x_train.columns
univariate.sort_values(ascending=False,inplace=True)

univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ =SelectKBest(f_classif,k=10).fit(x_train.fillna(0),y_train)
x_train.columns[sel_.get_support()]

x_train=sel_.transform(x_train.fillna(0))
x_train.shape


# REgression
data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\houseprice_train.csv",nrows=50000)  
data.shape
data.head()

numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['SalePrice'],axis=1),data['SalePrice'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape


univariate = f_regression(x_train.fillna(0),y_train)
univariate

univariate = pd.Series(univariate[1])
univariate.index=x_train.columns
univariate.sort_values(ascending=False,inplace=True)

univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ =SelectPercentile(f_regression,percentile=10).fit(x_train.fillna(0),y_train)
x_train.columns[sel_.get_support()]

x_train=sel_.transform(x_train.fillna(0))
x_train.shape




