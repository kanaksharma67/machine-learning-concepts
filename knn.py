import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as Plt
from sklearn.svm import SVC

iris = load_iris()

print(dir(iris))
print(iris.feature_names[0:5:1])

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df.target.apply(lambda x: iris.target[x])

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
X=df.drop(['target'],axis=1)
Y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("KNN Model Accuracy:",knn.score(X_test,y_test))
#Measure accuracy of your model using kernel such as rbf and linear
#Measure highest accuracy of your model using regulazrization and gamma parameters in svc()
print(knn.score(X_test,y_test))
print(df.head(10))

cm=confusion_matrix(y_test, knn.predict(X_test))
cr=classification_report(y_test, knn.predict(X_test))
print(cr)