from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

digits=load_digits()
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
# lr=LogisticRegression()
# lr.fit(X_train, y_train)
# print("Logistic Regression accuracy:", lr.score(X_test, y_test))




# kf=KFold(n_splits=3, shuffle=True)
# for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
#     print(train_index,test_index)
# [0 2 3 4 6 8] [1 5 7]
# [1 3 4 5 6 7] [0 2 8]
# [0 1 2 5 7 8] [3 4 6]



def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    print(f"First 10 predictions: {predictions[:10]}")  # Show first 10 predictions
    # print(f"Predictions: {predictions}")  # Show first 10 predictions
    print(f"{model.__class__.__name__} accuracy: {accuracy}")
    return accuracy



# run_model(LogisticRegression(), X_train, y_train, X_test, y_test)
# run_model(SVC(), X_train, y_train, X_test, y_test)
# run_model(RandomForestClassifier(), X_train, y_train, X_test, y_test)

folds=StratifiedKFold(n_splits=3, shuffle=True)
scores=[]
scores_svc=[]
scores_rf=[]

for train_index, test_index in folds.split(digits.data,digits.target):
    # digits.data[train_index], digits.data[test_index]
    X_train, X_test = digits.data[train_index], digits.data[test_index]
    y_train, y_test = digits.target[train_index], digits.target[test_index]



scores.append(("Logistic Regression", run_model(LogisticRegression(), X_train, y_train, X_test, y_test)))
scores_svc.append(("SVC", run_model(SVC(), X_train, y_train, X_test, y_test)))
scores_rf.append(("Random Forest", run_model(RandomForestClassifier(), X_train, y_train, X_test, y_test)))


print("Logistic Regression scores:", scores )
print("SVC scores:", scores_svc)
print("Random Forest scores:", scores_rf)



# crossval score

# model,X,Y
cross_val_scores= cross_val_score(LogisticRegression(), digits.data, digits.target, cv=folds)
print("Cross-validation scores for Logistic Regression:", cross_val_scores)