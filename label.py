import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df=pd.read_csv("carprices.csv")
le=LabelEncoder()
dfle=df
dfle["Car Model"]=le.fit_transform(dfle["Car Model"])


X=dfle[["Mileage", "Age(yrs)", "Car Model"]]

Y=dfle[["Sell Price($)"]]
ohe=OneHotEncoder()
ohe.fit_transform(X).toarray()
# print(X)


model=LinearRegression()
model.fit(X,Y)


X_train,Y_train , X_test, Y_test= train_test_split(X,Y, test_size=0.2,random_state=10)
# model.fit(X_train, Y_train)
model.fit(X_test, Y_test)
print(model.predict([[67000, 4, 1]]))
print(model.score(X,Y))

# accuracy rate is low as 87%  
# but by pandas it has 95%     