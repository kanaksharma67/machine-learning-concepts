import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import confusion_matrix
digits=load_digits()

print(dir(digits))
# print(digits.data[0])
for i in range(10):
    plt.figure(figsize=(3, 3))
    plt.imshow(digits.images[i], cmap="gray")  # imshow is slightly more common than matshow here
# plt.axis("off")                            # hide axes ticks
# plt.show()     


print(digits.target[0:5])

X_train, X_test, Y_train, Y_test= train_test_split(digits.data, digits.target,test_size=0.2)
model=LogisticRegression()
model.fit(X_train, Y_train)
plt.imshow(digits.images[67], cmap='gray')
# print(model.predict([[digits.data[67]]]))
print(model.predict([digits.data[5]]))  # ✅ CORRECT


print(model.score(X_test, Y_test))

y_predicted=model.predict(X_test)
cm=confusion_matrix(Y_test, y_predicted)
print(cm)



with open("number_prediction", "wb") as f:
    pickle.dump(model,f)

with open("number_prediction", "rb") as f:
    mp=pickle.load(f)

print(mp)