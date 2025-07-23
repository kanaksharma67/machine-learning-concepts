from sklearn.ensemble import IsolationForest
import pandas as pd

df=pd.Dataframe({'value':[2,3,4,100,1,3,2,5,2]})
X = data[['value']]
iso = IsolationForest(contamination=0.1)  # 10% outliers
yhat = iso.fit_predict(X)
filtered_data = data[yhat == 1]
print(filtered_data)
