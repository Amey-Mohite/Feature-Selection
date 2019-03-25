import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
 
from sklearn.model_selection import train_test_split
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score

data = pd.read_csv('G:\study\machine learning\udemy\Feature Selection\paribas_train.csv', nrows=50000)
data.shape

data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['target', 'ID'], axis=1),
    data['target'],
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape

sel_ = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10)
sel_.fit(X_train.fillna(0), y_train)

sel_.get_support()

selected_feat = X_train.columns[(sel_.get_support())]
len(selected_feat)

selected_feat

previous_lecture_selected_features = [
    'v10', 'v12', 'v14', 'v21', 'v34', 'v40', 'v50', 'v62', 'v114', 'v129'
]



def run_randomForests(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

# features selected recursively
run_randomForests(X_train[selected_feat].fillna(0),
                  X_test[selected_feat].fillna(0),
                  y_train, y_test)

# features selected altogether
run_randomForests(X_train[previous_lecture_selected_features].fillna(0),
                  X_test[previous_lecture_selected_features].fillna(0),
                  y_train, y_test)

