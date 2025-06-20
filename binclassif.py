import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49],
    "insurance": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

plt.scatter(df["age"], df["insurance"], color="blue", marker='+')
# plt.show()

# Correct order of variables
X_train, X_test, Y_train, Y_test = train_test_split(df[['age']], df[['insurance']], test_size=0.1)

model = LogisticRegression()
model.fit(X_train, Y_train)
print(X_test)
print(model.predict(X_test))
print(model.predict([[25]]))

print(model.predict_proba(X_test))

print(model.score(df[["age"]], df[["insurance"]]))