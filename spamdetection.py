import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df=pd.read_csv("spam.csv", encoding="latin-1")
print(df.head())
des=df.groupby("Category").describe()
print(des)

dfle=LabelEncoder()

df["Category_n"]=dfle.fit_transform(df["Category"])



X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Category_n"], test_size=0.2, random_state=42)

folds=StratifiedKFold(n_splits=3, shuffle=True)
for train_index, test_index in folds.split(df["Message"], df["Category_n"]):
    X_train, X_test = df["Message"].values[train_index], df["Message"].values[test_index]
    y_train, y_test = df["Category_n"].values[train_index], df["Category_n"].values[test_index]

V=CountVectorizer()
X_train_count=V.fit_transform(X_train.values)
X_train_count=X_train_count.toarray()# Convert sparse matrix to dense array used for message text

model=MultinomialNB()
model.fit(X_train_count, y_train)


emails=["Free entry in 2 a weekly competition to win FA Cup tickets. Text FA to 87121 to receive entry question (cost 1.50) and receive entry into the draw. T&Cs apply. 08452810075over18s",
         "Hi, how are you? I hope you are doing well. Let's catch up soon!"]
emails_count=V.transform(emails)
print(model.predict(emails_count))
print(model.score(X_train_count, y_train))



clf=Pipeline([("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())])
clf.fit(X_train, y_train)
print(clf.predict(emails))
print(clf.score(X_test, y_test))