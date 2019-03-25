import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score,mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000)
data.shape

numeric=['int16','int32','int64','float16','float32','float64']
numeric_vars=list(data.select_dtypes(include=numeric).columns)
data=data[numeric_vars]
data.shape


X_train,X_test,y_train,y_test=tts(data.drop(labels=['target','ID'],axis=1),data['target'],test_size=0.3,random_state=0)
X_train.shape,X_test.shape

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
         for j in range(i):
             if abs(corr_matrix.iloc[i,j])>threshold:
                 colname=corr_matrix.columns[i]
                 col_corr.add(colname)
    return col_corr             
                 
corr_features=correlation(X_train,0.8)
print('correlated features:',len(set(corr_features)))

X_train.drop(labels=corr_features,axis=1,inplace=True)
X_test.drop(labels=corr_features,axis=1,inplace=True)

efs1=EFS(RandomForestClassifier(n_jobs=4,random_state=0),
         min_features=1 ,
         max_features=4,
         scoring='roc_auc',
         print_progress = True,
         cv=2
         )

efs1=efs1.fit(np.array(X_train[X_train.columns[0:4]].fillna(0)),y_train)
select_feat= X_train.columns[list(efs1.best_idx_)]
select_feat

def run_randomForests(X_train,X_test,y_train,y_test):
    rf=RandomForestClassifier(n_estimators=200,random_state=39,max_depth=4)
    rf.fit(X_train,y_train)
    print('Train set')
    pred=rf.predict_proba(X_train)
    print('Random Forests roc_auc :{}'.format(roc_auc_score(y_train,pred[:,1])))
    print('Test set')
    pred=rf.predict_proba(X_test)
    print('Random Forests roc_auc :{}'.format(roc_auc_score(y_test,pred[:,1])))
    
run_randomForests(X_train[select_feat].fillna(0),X_test[select_feat].fillna(0),y_train,y_test)    

####################################
#Regression
data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\houseprice_train.csv")
data.shape

numeric=['int16','int32','int64','float16','float32','float64']
numeric_vars=list(data.select_dtypes(include=numeric).columns)
data=data[numeric_vars]
data.shape


X_train,X_test,y_train,y_test=tts(data.drop(labels=['SalePrice'],axis=1),data['SalePrice'],test_size=0.3,random_state=0)
X_train.shape,X_test.shape

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
         for j in range(i):
             if abs(corr_matrix.iloc[i,j])>threshold:
                 colname=corr_matrix.columns[i]
                 col_corr.add(colname)
    return col_corr             
                 
corr_features=correlation(X_train,0.8)
print('correlated features:',len(set(corr_features)))

X_train.drop(labels=corr_features,axis=1,inplace=True)
X_test.drop(labels=corr_features,axis=1,inplace=True)

efs1=EFS(RandomForestRegressor(n_jobs=4),
         min_features=1,
         max_features=4,
         scoring='r2',
         print_progress=True,
         cv=2
         )

efs1=efs1.fit(np.array(X_train[X_train.columns[0:4]].fillna(0)),y_train)
select_feat= X_train.columns[list(efs1.best_idx_)]
select_feat

def run_randomForests(X_train,X_test,y_train,y_test):
    rf=RandomForestRegressor(n_estimators=200,random_state=39,max_depth=4)
    rf.fit(X_train,y_train)
    print('Train set')
    pred=rf.predict(X_train)
    print('Random Forests r2 :{}'.format(mean_squared_error(y_train,pred)))
    print('Tesst set')
    pred=rf.predict(X_test)
    print('Random Forests r2 :{}'.format(mean_squared_error(y_test,pred)))
    
run_randomForests(X_train[select_feat].fillna(0),X_test[select_feat].fillna(0),y_train,y_test)







