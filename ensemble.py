import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
df=pd.read_csv("diabetes.csv")
print(df.head())

print(df.isnull().sum())
print(df.describe())

# there are no outliers
print(df.Outcome.value_counts())

X=df.drop('Outcome',axis=1)
Y=df['Outcome']

scaler=StandardScaler()
X_s=scaler.fit_transform(X)


X_train,X_test,Y_train,Y_test=train_test_split(X_s,Y,test_size=0.2,stratify=Y,random_state=10)

scores=cross_val_score(DecisionTreeClassifier(),X,Y,cv=5)
# oob=out of bag means some data set like 29 will not be in M then that will bw test data set , n_estimators means how many M will be their from our dataset , max_sample means how many will be train dataset, Base_estimators means the model you will use the algorithm
bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=0.8,oob_score=True, random_state=0)
bag_model.fit(X,Y)
print(bag_model.oob_score_)
scores=cross_val_score(bag_model,X,Y,cv=5)
print(scores.mean())




# <-------------THE END :) --------------->