import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


digits=load_digits()
digits.data[0].reshape(8, 8)
plt.matshow(digits.data[0].reshape(8, 8), cmap='gray')
# plt.show()
un=np.unique(digits.target)
print(un)

df=pd.DataFrame(digits.data,columns=digits.feature_names)
df['target']=digits.target
print(df.head())
X=df.drop('target',axis=1)
Y=df['target']


scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled)

X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,test_size=0.2, random_state=30)


model=LogisticRegression()
model.fit(X_train,Y_train)
print("Model Accuracy:", model.score(X_test, Y_test))

# pca for dimensionality reduction
pca=PCA(n_components=0.95)#retain 95% of variance i.e useful features
X_pca=pca.fit_transform(X)
print(X_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.components_)
X_train,X_test,Y_train,Y_test=train_test_split(X_pca,Y,test_size=0.2, random_state=30)
model=LogisticRegression(n_itter=1000, tol=0.0001)
model.fit(X_train,Y_train)
print("Model Accuracy after PCA:", model.score(X_test, Y_test))
