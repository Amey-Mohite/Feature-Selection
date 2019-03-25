import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


#for small data
data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\santander_train.csv",nrows=15000)
data.shape

[col for col in data.columns if data[col].isnull().sum()>0]

x_train,x_test,y_train,y_test=train_test_split(data.drop(labels=["TARGET"],axis=1),data["TARGET"],test_size=0.3,random_state=0)
x_train.shape
x_test.shape  

data_t=x_train.T

data_t.head()

data_t.duplicated().sum()

data_t[data_t.duplicated()]

duplicated_features=data_t[data_t.duplicated()].index.values

duplicated_features

data_unique= data_t.drop_duplicates(keep='first').T

data_unique.shape

duplicated_featured=[col for col in data.columns if col not in data_unique.columns]
duplicated_featured



# if datasets is big 

data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\santander_train.csv")
data.shape

[col for col in data.columns if data[col].isnull().sum()>0]

x_train,x_test,y_train,y_test=train_test_split(data.drop(labels=["TARGET"],axis=1),data["TARGET"],test_size=0.3,random_state=0)
x_train.shape
x_test.shape  

duplicate_feat=[]
for i in range(0,len(x_train.columns)):
    if i % 10 == 0:
        print(i)
    
    col_1=x_train.columns[i]
    for col_2 in x_train.columns[i+1:]:
        if x_train[col_1].equals(x_train[col_2]):
            duplicate_feat.append(col_2)
            
        
print(len(set(duplicate_feat)))

set(duplicate_feat)


duplicated_feat=[]
for i in range(0,len(x_train.columns)):
    col_1=x_train.columns[i]
    for col_2 in x_train.columns[i+1:]:
        if x_train[col_1].equals(x_train[col_2]):
            print(col_1)
            print(col_2)
            print()
            
            duplicated_feat.append(col_2)
            


