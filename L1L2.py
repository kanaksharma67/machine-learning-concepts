import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

data = pd.read_csv("Melbourne_housing_FULL (1).csv")

# Get unique values across the whole DataFrame
data_unique = pd.unique(data.values.ravel())
print(data_unique[:10])
print("Total unique values:", len(data_unique))
print("Original shape:", data.shape)#(34857, 21)
print(data.columns)
cols_to_use=['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',      
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount']

print(data.isnull().sum())

cols_to_fill_zero=['Distance','Postcode','Bedroom2','Bathroom','Lattitude','Longtitude','Car',]
data[cols_to_fill_zero]=data[cols_to_fill_zero].fillna(0)
print(data.isnull().sum())
data['Landsize']=data['Landsize'].fillna(data['Landsize'].mean())
data['BuildingArea']=data['BuildingArea'].fillna(data['BuildingArea'].mean())
data['YearBuilt']=data['YearBuilt'].fillna(data['YearBuilt'].mean())
data.dropna(inplace=True)
print(data.isnull().sum())



data=pd.get_dummies(data,columns=['Suburb', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname'],drop_first=True)
X=data.drop('Price',axis=1)
Y=data['Price']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
reg= LinearRegression()
reg.fit(X_train, Y_train)
print("Training score:", reg.score(X_train, Y_train))


lesso_reg=linear_model.Lasso(alpha=0.1)
lesso_reg.fit(X_train,Y_train)
print("Lasso Training score:", lesso_reg.score(X_train, Y_train))
print("Lasso Test score:", lesso_reg.score(X_test, Y_test))


ridge_reg=linear_model.Ridge(alpha=0.1)
ridge_reg.fit(X_train,Y_train)
print("Ridge Training score:", ridge_reg.score(X_train, Y_train))
print("Ridge Test score:", ridge_reg.score(X_test, Y_test))