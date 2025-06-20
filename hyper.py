import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import svm 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower']=iris.target
df['flower']=df['flower'].apply(lambda x:iris.target_names[x])
# df['flower']=df['flower'].map('0: setosa, 1: versicolor, 2: virginica'.split(', '))
print(df.head(50))


X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target,test_size=0.2)


model=svm.SVC(kernel='rbf',C=30, gamma="auto")
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))


cross_val_scoress=cross_val_score(svm.SVC(kernel='rbf',C=30,gamma='auto'),iris.data,iris.target,cv=5)


clf=GridSearchCV(svm.SVC(),param_grid={'C':[1,10,100],'gamma':[0.001,0.01,0.1,1]},cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)
# print(clf.score())
print("clf.cv_results_:", clf.cv_results_)
df=pd.DataFrame(clf.cv_results_)
df=pd.to_csv("grid_search_results.csv", index=False)
print(df[['param_C', 'param_gamma', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False))


rs=RandomizedSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,100],
    'kernel':['linear','rbf','poly']
},
    n_iter=10, cv=5, return_train_score=False
 )

rs.fit(iris.data,iris.target)
pd.DataFrame(rs.cv_results_)













# Choosing best model and hyperparameters
model_params={
    "svm": {
        "model": svm.SVC(gamma='auto'),
        "params": {
            'C': [1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30]
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(),
        "params": {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    }
}
scores=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })  

df=pd.DataFrame(scores,columns=['model','best_score','best_params'])
