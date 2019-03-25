import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts

data=pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\paribas_train.csv",nrows=50000)
data.shape
data.head()

numerics=['int16','int32','int64','float16','float32','float64']
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape

x_train,x_test,y_train,y_test=tts(data.drop(labels=['target','ID'],axis=1),data['target'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape

#correlatin method
corrmat=x_train.corr()
fig,ax=plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)

# coding method

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr            

corr_features=correlation(x_train,0.8)                
len(set(corr_features))

x_train.drop(labels=corr_features,axis=1,inplace=True)
x_test.drop(labels=corr_features,axis=1,inplace=True)


#another method
 
corrmat=x_train.corr()
corrmat=corrmat.abs().unstack()
corrmat= corrmat.sort_values(ascending=False)
corrmat=corrmat[corrmat>=0.8]
corrmat=corrmat[corrmat<1]
corrmat=pd.DataFrame(corrmat).reset_index()
corrmat.columns=['feature1','feature2','corr']
corrmat.head()

group_feature_ls=[]
correlated_groups=[]

for feature in corrmat.feature1.unique():
    if feature not in group_feature_ls:
        correlated_blocks=corrmat[corrmat.feature1==feature]
        group_feature_ls=group_feature_ls+list(correlated_blocks.feature2.unique())+[feature]
        correlated_groups.append(correlated_blocks)
print('found {} correlated_groups'.format(len(correlated_groups)))
print('out of {} total features'.format(x_train.shape[1]))


for group in correlated_groups:
    print(group)
    print()

group = correlated_groups[2]
group 



for feature in list(group.feature2.unique())+['v17']:
    print(x_train[feature].isnull().sum())



from sklearn.ensemble import RandomForestClassifier

features= list(group.feature2.unique())+['v17']
rf=RandomForestClassifier(n_estimators=200,random_state=39,max_depth=4)
rf.fit(x_train[features].fillna(0),y_train)
importance = pd.concat([pd.Series(features),pd.Series(rf.feature_importances_)],axis=1)

importance.columns =['features','importance']
importance.sort_values(by='importance',ascending=False)
