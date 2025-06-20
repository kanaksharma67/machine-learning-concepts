# for SINGLE VARIABLE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import joblib


data={
    "area":[2600,3000,3200,3600,4000],
    "price":[550000,565000, 610000,680000,725000]
}
df=pd.DataFrame(data)

reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df[["price"]])
# print(reg.predict([[3300]]
plt.scatter(df["area"],df["price"], color='cyan', marker='o')
plt.plot(df["area"],reg.predict(df[["area"]]),color='blue')

plt.xlabel("Area ")
plt.ylabel("Prices ")

# plt.show()

#rb read in binary mode

with open("model_pickle", "wb") as f :
    pickle.dump(reg,f)
with open("model_pickle", "rb") as f:
    mp=pickle.load(f)

print(mp.predict([[3000]]))


# print(reg.coef_)#===m means slope predicted 
# print(reg.intercept_)#== b means intercept value 

# # y=m*x+b  here y =prices
# print(135.7876712*628715.75342466+180616.43835616 )


# import pandas as pd
# from sklearn import linear_model

# # Training data
# data = {
#     "area": [2600, 3000, 3200, 3600, 4000],
#     "price": [550000, 565000, 610000, 680000, 725000]
# }
# df = pd.DataFrame(data)

# # Train the model
# reg = linear_model.LinearRegression()
# reg.fit(df[["area"]], df[["price"]])

# # New data to predict
# dataarea = {
#     "area": [1000, 1500, 2300, 3540, 4120, 4560, 5490, 3460]
# }
# dfe = pd.DataFrame(dataarea)

# # Predict price for new areas
# dfe["predicted_price"] = reg.predict(dfe[["area"]])

# # Save to CSV
# dfe.to_csv("output.csv", index=False)






# save model using sklearn joblib we use jobglib when my data has large nuber of numpy arrays
# search sklearn model presistence in google
joblib.dump(reg, "model_joblib")
mj=joblib.load("model_joblib")
print(mj.predict([[4000]]))



