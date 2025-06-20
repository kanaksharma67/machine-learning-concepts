# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression



# lr=load_iris()
# print(dir(lr))
# print(lr.data[0])

# X_train, X_test, Y_train, Y_test=train_test_split(lr.data,lr.target)
# model=LogisticRegression()
# model.fit(X_train, Y_train)
# print(model.predict([lr.data[20]]))  # ✅ CORRECT


# print(model.score(X_test, Y_test))
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as Plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

print(dir(iris))
print(iris.feature_names[0:5:1])

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

Plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='blue', label='Setosa')
Plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='green', label='Versicolor')
Plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', label='Virginica')
Plt.xlabel("Sepal Length (cm)")
Plt.ylabel("Sepal Width (cm)")
Plt.legend()
Plt.show()

X = df.drop(['target', 'flower_name'], axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = SVC()
model.fit(x_train, y_train)
print("Model Accuracy:", model.score(x_test, y_test))



# measure accuracy of your model using kernel such as rbf and linear
# measure highest accuracy of your model using regulazrization and gamma parameters in svc()
