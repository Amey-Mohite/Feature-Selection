import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\santander_train.csv")
data.shape

[col for col in data.columns if data[col].isnull().sum()>0]

x_train,x_test,y_train,y_test=train_test_split(data.drop(labels=["TARGET"],axis=1),data["TARGET"],test_size=0.3,random_state=0)
x_train.shape
x_test.shape

#variance Threshhold
sel=VarianceThreshold(threshold=0)
sel.fit(x_train)
sum(sel.get_support())
#another way
len(x_train.columns[sel.get_support()])

print(
      len([
           x for x in x_train.columns
           if x not in x_train.columns[sel.get_support()]
              ]))
 
[x for x in x_train.columns if x not in x_train.columns[sel.get_support()]]

x_train['ind_var2_0'].unique()

x_train=sel.transform(x_train)
x_test=sel.transform(x_test)


#short adn easy

constant_features=[feat for feat in x_train.columns if x_train[feat].std()==0]
len(constant_features)

x_train.drop(labels=constant_features,axis=1, inplace=True)

x_test.drop(labels=constant_features,axis=1, inplace=True)


#for categorical Variables

x_train=x_train.astype("O")
x_train.dtypes
constant_features=[feat for feat in x_train.columns if len(x_train[feat].unique())==1]
len(constant_features)
 