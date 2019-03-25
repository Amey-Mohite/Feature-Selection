import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

#data Reading 
data = pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\santander_train.csv")
data.shape

# Train And Test
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['TARGET'], axis=1),
    data['TARGET'],
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape

# Copying Original Data
X_train_original = X_train.copy()
X_test_original = X_test.copy()

#removing constant Features

constant_features=[col for col in X_train.columns if X_train[col].std()==0]
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)
 
X_train.shape, X_test.shape



#removing QuansiConstant Features
sel =VarianceThreshold(threshold=0.01)
sel.fit(X_train)
sum(sel.get_support())
features_to_keep = X_train.columns[sel.get_support()]

X_train=sel.transform(X_train)

X_test=sel.transform(X_test)

X_train=pd.DataFrame(X_train)
X_train.columns=features_to_keep
X_test=pd.DataFrame(X_test)
X_test.columns=features_to_keep


#removing Duplicate Columns
duplicated_feat=[]
for i in range(0,len(X_train.columns)):
    if i%10==0:
        print(i)
    col_1=X_train.columns[i]
    
    for col_2 in X_train.columns[i+1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
len(duplicated_feat)

X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)
 
X_train.shape, X_test.shape

#filter till Basic Filter
X_train_basic_filter = X_train.copy()
X_test_basic_filter = X_test.copy()

#correlation
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr            
corr_features=correlation(X_train,0.8)
print('correlated features: ', len(set(corr_features)) )

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
 
X_train.shape, X_test.shape

X_train_corr = X_train.copy()
X_test_corr = X_test.copy()

#Using ROC_AUC
roc_values = []
for feature in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))


roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))

selected_feat = roc_values[roc_values>0.5]
len(selected_feat), X_train.shape[1]

def run_randomForests(X_train,X_test,y_train,y_test):
    rf=RandomForestClassifier(n_estimators=200,random_state=39,max_depth=4)
    rf.fit(X_train,y_train)
    print("train Set")
    pred=rf.predict_proba(X_train)
    print('Random Forest roc-aoc:{}'.format(roc_auc_score(y_train,pred[:,1])))
    print('test Set')
    pred=rf.predict_proba(X_test)
    print('Random Forest roc-aoc:{}'.format(roc_auc_score(y_test,pred[:,1])))

#original
run_randomForests(X_train_original.drop(labels=['ID'], axis=1),
                  X_test_original.drop(labels=['ID'], axis=1),
                  y_train, y_test)  

# filter methods - basic
run_randomForests(X_train_basic_filter.drop(labels=['ID'], axis=1),
                  X_test_basic_filter.drop(labels=['ID'], axis=1),
                  y_train, y_test)


# filter methods - correlation
run_randomForests(X_train_corr.drop(labels=['ID'], axis=1),
                  X_test_corr.drop(labels=['ID'], axis=1),
                  y_train, y_test)


# filter methods - univariate roc-auc
run_randomForests(X_train[selected_feat.index],
                  X_test_corr[selected_feat.index],
                  y_train, y_test)

def run_logistic(X_train,X_test,y_train,y_test):
    logit=LogisticRegression(random_state=44)
    logit.fit(X_train,y_train)
    print("train Set")
    pred=logit.predict_proba(X_train)
    print('Random Forest roc-aoc:{}'.format(roc_auc_score(y_train,pred[:,1])))
    print('test Set')
    pred=logit.predict_proba(X_test)
    print('Random Forest roc-aoc:{}'.format(roc_auc_score(y_test,pred[:,1])))

#original
scaler = StandardScaler().fit(X_train_original.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train_original.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test_original.drop(labels=['ID'], axis=1)),
                  y_train, y_test) 

# filter methods - basic
scaler = StandardScaler().fit(X_train_basic_filter.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train_basic_filter.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test_basic_filter.drop(labels=['ID'], axis=1)),
                  y_train, y_test)

# filter methods - correlation
scaler = StandardScaler().fit(X_train_corr.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train_corr.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test_corr.drop(labels=['ID'], axis=1)),
                  y_train, y_test)

  
# filter methods - univariate roc-auc
scaler = StandardScaler().fit(X_train[selected_feat.index])
 
run_logistic(scaler.transform(X_train[selected_feat.index]),
             scaler.transform(X_test_corr[selected_feat.index]),
                  y_train, y_test)


























