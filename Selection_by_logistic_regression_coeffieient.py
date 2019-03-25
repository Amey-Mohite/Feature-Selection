import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

data= pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000)
data.shape

data.head()

numeric=['int16','int32','int64','float16','float32','float64']
numeric_vars=list(data.select_dtypes(include=numeric).columns)
data=data[numeric_vars]
data.shape

X_train,X_test,y_train,y_test=tts(data.drop(labels=['target','ID'],axis=1),data['target'],test_size=0.3,random_state=0)
X_train.shape
X_test.shape

scaler=StandardScaler()
scaler.fit(X_train.fillna(0))

sel_=SelectFromModel(LogisticRegression(C=1000,penalty='l1'))
sel_.fit(scaler.transform(X_train.fillna(0)),y_train)

sel_.get_support()
selected_feat= X_train.columns[(sel_.get_support())]
len(selected_feat)

np.sum(sel_.estimator_.coef_ == 0)

sel_.estimator_.coef_.mean()

pd.Series(sel_.estimator_.coef_.ravel()).hist()

np.abs(sel_.estimator_.coef_.mean())

pd.Series(np.abs(sel_.estimator_.coef_).ravel()).hist()


print('total feature : {}'.format((X_train.shape[1])))
print('selected features'.format(len(selected_feat)))
print('features with coefficient greater than the mean difference: {}'.format(np.sum(np.abs(sel_.estimator_.coef_)>np.abs(sel_.estimator_.coef_).mean())))


























