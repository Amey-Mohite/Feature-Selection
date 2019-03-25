import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score,mean_squared_error

data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000)  
data.shape
data.head()

numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['target','ID'],axis=1),data['target'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape


roc_values=[]
for feature in x_train.columns:
    clf=DecisionTreeClassifier()
    clf.fit(x_train[feature].fillna(0).to_frame(),y_train)
    y_scored=clf.predict_proba(x_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test,y_scored[:,1]))
    
roc_values=pd.Series(roc_values)
roc_values.index=x_train.columns    
roc_values.sort_values(ascending=False)

roc_values.sort_values(ascending=False).plot.bar(figsize=(20,8))

len(roc_values[roc_values>0.5])



#Regression

data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\houseprice_train.csv",nrows=50000)  
data.shape
data.head()

numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['SalePrice'],axis=1),data['SalePrice'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape

mse_values=[]
for feature in x_train.columns:
    clf=DecisionTreeRegressor()
    clf.fit(x_train[feature].fillna(0).to_frame(),y_train)
    y_scored=clf.predict(x_test[feature].fillna(0).to_frame())
    mse_values.append(mean_squared_error(y_test,y_scored))

mse_values=pd.Series(mse_values)
mse_values.index=x_train.columns    
mse_values.sort_values(ascending=False)

mse_values.sort_values(ascending=False).plot.bar(figsize=(20,8))



