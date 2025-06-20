import pandas as pd
from sklearn.linear_model import LinearRegression



# data={
#     "Town":["Monroe","Monroe","Monroe" "west","west","west", "robin","robin","robin"],
#     "area":[2600,2500,1000,4000,3000,1000,3800,2500,7300],
#     "price":[55000,32000,10100,23000,23200,12000,34900,95400,54000]
# }

df=pd.read_csv("carprices.csv")



# get the one hot encoding ,et6hod tabel
dummy=pd.get_dummies(df["Car Model"])
print(dummy)


#merge the df and dummy data tabel together by 
merged=pd.concat([df,dummy],axis=1)
# print(merged)

#drop the car model col as its in string we dont want that now   as we are using sklearn it eill directly drop the col but its good practice to drop the column
merged=merged.drop(["Car Model"],axis=1)
print(merged)

model=LinearRegression()
model.fit(merged[["Mileage", "Age(yrs)","Audi A5", "BMW X5", "Mercedez Benz C class"]], merged[["Sell Price($)"]])

print(model.predict([[45000,3, True,False, False]]))
print(model.predict([[34000,2, False, True, False]]))


print(model.score(merged[["Mileage", "Age(yrs)","Audi A5", "BMW X5", "Mercedez Benz C class"]],merged[["Sell Price($)"]])) #if 1. somethng that means your model is  100%accurate
