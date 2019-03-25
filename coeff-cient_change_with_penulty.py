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
 
coef_df= []
for c in [1,10,100,1000]:
    logit=LogisticRegression(C=c,penalty='l2')
    logit.fit(scaler.transform(X_train.fillna(0)),y_train)
    coef_df.append(pd.Series(logit.coef_.ravel()))
    
coef = pd.concat(coef_df,axis=1)
coef.columns = [1,10,100,1000]
coef.index = X_train.columns
coef.head()    



coef.columns = np.log([1,10,100,1000])
coef.head()

coef.T.plot(figsize=(15,10))

temp=coef.head(10)
temp=temp.T
temp.plot(figsize=(12,8))


