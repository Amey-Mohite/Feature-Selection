import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest,SelectPercentile

data =pd.read_csv(r"G:\study\machine learning\udemy\Feature Selection\titanic_train.csv")  
data.shape
data.head()

data['Sex']=np.where(data.Sex=='male',1,0)
ordinal_label={k:i for i,k in enumerate(data['Embarked'].unique(),0)}
data['Embarked']=data['Embarked'].map(ordinal_label)

x_train,x_test,y_train,y_test=tts(data[['Pclass','Sex','Embarked']],data['Survived'],test_size=0.3,random_state=0)
x_train.shape,x_test.shape

f_score=chi2(x_train.fillna(0),y_train)
f_score
pvalues=pd.Series(f_score[1])
pvalues.index=x_train.columns
pvalues.sort_values(ascending=False)
