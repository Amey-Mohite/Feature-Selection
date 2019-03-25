import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\santander_train.csv",nrows=50000)
data.shape

[col for col in data.columns if data[col].isnull().sum()>0]

x_train,x_test,y_train,y_test=train_test_split(data.drop(labels=["TARGET"],axis=1),data["TARGET"],test_size=0.3,random_state=0)
x_train.shape
x_test.shape

constant_features=[feat for feat in x_train.columns if x_train[feat].std()==0]
x_train.drop(labels=constant_features,axis=1,inplace=True)
x_test.drop(labels=constant_features,axis=1,inplace=True)

x_train.shape
x_test.shape

sel = VarianceThreshold(threshold=0.01)
sel.fit(x_train)
sum(sel.get_support())


print(
      len([
          x for x in x_train.columns
          if x not in x_train.columns[sel.get_support()]
              ])
      )

[x for x in x_train.columns if x not in x_train.columns[sel.get_support()]]
x_train['ind_var31'].value_counts()/np.float(len(x_train))

x_train =sel.transform(x_train)
x_test= sel.transform(x_test)
x_train.shape,x_test.shape


#by coding 

quasi_constant_feat=[]
for feature in x_train.columns:
    predominant=(x_train[feature].value_counts()/np.float(len(x_train))).sort_values(ascending=False).values[0]
    
    if predominant>0.998:
        quasi_constant_feat.append(feature)

len(quasi_constant_feat)
(x_train[feature].value_counts()/np.float(len(x_train))).sort_values(ascending=False)














