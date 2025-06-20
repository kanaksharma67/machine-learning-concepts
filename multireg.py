import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model


data={
    "area":[2500,3000,3200,3600,4000],
    "bedrooms":[3,4,None,3,5],
    "age":[20,15,18,30,8],
    "price":[550000,565000,610000,595000,760000]
}

df=pd.DataFrame(data)
med=math.floor(df["bedrooms"].median())
print(med)
df["bedrooms"]=df["bedrooms"].fillna(med)


# print(df)

reg=linear_model.LinearRegression()
# reg.fit("independent variable", "target variable")
reg.fit(df[["bedrooms", "area","age"]], df[["price"]])

print(reg.coef_)#will give m1 ,m2 and m3
print(reg.intercept_)#value of b

print(reg.predict([[3,3000,40]]))
print(reg.predict([[3,3000,40]]))
# y=m1x1+m2x2+m3x3+b
# to convert string to number we will use pip install word2number
