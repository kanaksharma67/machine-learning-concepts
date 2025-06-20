from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# data={
#     "area":[27, 29, 29, 28, 42, 39, 41, 38, 36, 26, 27,28, 29,32,40,41,43,39,41,39],
#     "income":[7000,9000,6100,1500,1550,1600,1620,1560,1300,1370,4500,4800,5100,4900,5300,6500,6300,6400,8000,8200]
# }

# df=pd.DataFrame(data)


# plt.scatter(df["area"], df["income"], color="blue", label="Data Points")
# plt.legend()
# plt.xlabel("Area")
# plt.ylabel("Income")                              
# # plt.show()



# kc=KMeans(n_clusters=3)
# y_predicted=kc.fit_predict(df)  # Fit the model using only the area feature
# df["cluter"]=y_predicted

# df1=df[df["cluter"]==0]
# df2=df[df["cluter"]==1]
# df3=df[df["cluter"]==2]

# plt.scatter(df1["area"], df1["income"], color="red", label="Cluster 1")
# plt.scatter(df2["area"], df2["income"], color="green", label="Cluster 2")
# plt.scatter(df3["area"], df3["income"], color="blue", label="Cluster 3")
# # plt.show()

# scaler=MinMaxScaler()
# df_scaled=scaler.fit_transform(df[["area", "income"]])
# print(df_scaled)

# km=KMeans(n_clusters=3)
# y_predicted2=km.fit_predict(df_scaled)
# print(km.cluster_centers_)

# k_rng=range(1, 10)
# sse=[]
# for k in k_rng:
#     km=KMeans(n_clusters=k)
#     km.fit(df_scaled)
#     sse.append(km.inertia_)
# print(sse)
# plt.xlabel("K")
# plt.ylabel("Sum of Squared Errors")
# plt.plot(k_rng, sse)
# plt.show()
#  print(y_predicted2)


