# Titanic survival rate

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("titanic.csv")

# Drop unused columns
df = df.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)

# Target and input
target = df["Survived"]
input = df.drop("Survived", axis=1)

# Handle categorical column 'Sex'
dummies = pd.get_dummies(input["Sex"])
input_n = pd.concat([input, dummies], axis=1)
input_n = input_n.drop("Sex", axis=1)

# Handle missing values
print(input_n.isnull().sum())
input_n["Age"].fillna(input_n["Age"].mean(), inplace=True)
print(input_n.isnull().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(input_n, target, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Accuracy
print("Training accuracy:", model.score(X_test, y_test))
# Prediction 
print("Prediction:", model.predict_proba(X_test[:10]))