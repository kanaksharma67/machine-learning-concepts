# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn import tree

# data = {
#     "company": ["google", "google", "google", "google", "google", "google",
#                 "Facebook", "Facebook", "Facebook", "Facebook", "Facebook",
#                 "abc", "abc", "abc", "abc", "abc"],
#     "job": ["sales", "sales", "business", "business", "cs", "cs",
#             "sales", "cs", "business", "business", "sales",
#             "sales", "business", "business", "cs", "cs"],
#     "degree": ["B", "M", "B", "M", "B", "M", "M", "B", "B", "M",
#                "B", "M", "B", "M", "B", "M"],
#     "salary": [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# }

# df = pd.DataFrame(data)

# inputs = df.drop('salary', axis=1)
# target = df["salary"]

# # Label encoding correctly
# le_company = LabelEncoder()
# le_job = LabelEncoder()
# le_degree = LabelEncoder()

# inputs["company_n"] = le_company.fit_transform(inputs["company"])
# inputs["job_n"] = le_job.fit_transform(inputs["job"])
# inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])

# # Drop original text columns
# inputs_n = inputs.drop(["company", "job", "degree"], axis=1)


# # Train Decision Tree model   
# model = tree.DecisionTreeClassifier()   
# model.fit(inputs_n, target)   
# print(inputs_n)   
# # Accuracy (on training data)   
# print("Training accuracy:", model.score(inputs_n, target))    


# print("prediction:", model.predict([["0","1", "0"]]))


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree



df=pd.read_csv("titanic.csv")
input=df.drop("Survived", axis=1)
target=df["Survived"]

lename=LabelEncoder()  
lesex=LabelEncoder()  
letic=LabelEncoder()  
lecabin=LabelEncoder()   
leembarked=LabelEncoder()    

input["Name_n"]=lename.fit_transform(input["Name"])
input["sex_n"]=lesex.fit_transform(input["Sex"])
input["ticket_n"]=letic.fit_transform(input["Ticket"])
input["cabin_n"]=lecabin.fit_transform(input["Cabin"])
input["embarked_n"]=leembarked.fit_transform(input["Embarked"])

input_n=input.drop(['Name', "Sex", "Ticket","Cabin","Embarked"], axis=1)

model=tree.DecisionTreeClassifier()  

model.fit(input_n, target)
print(model.score(input_n, target))

print(input_n)
print(model.predict)