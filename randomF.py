import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier#ensemble is used when you use multiple algorithm
import pandas as pd
from sklearn.metrics import confusion_matrix



digits=load_digits()

print(dir(digits))    
# print(digits.data[0])  
for i in range(10):   
    plt.figure(figsize=(3, 3))      
    plt.imshow(digits.images[i], cmap="gray")  # imshow is slightly more common than matshow here
# plt.axis("off")                            # hide axes ticks
# plt.show()     


df=pd.DataFrame(digits.data)    
df["target"]=digits.target    
print(df.head())               

x_train,x_test, y_train, y_test=train_test_split(df.drop("target", axis=1), df.target, test_size=0.2)
model=RandomForestClassifier(n_estimators=100)#10 random tree
model.fit(x_train, y_train)      
print(model.score(x_test, y_test))     
y_predicted=model.predict(x_test)     
cm=confusion_matrix(y_test, y_predicted)     
print(cm)    

