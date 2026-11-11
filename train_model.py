from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib
iris = load_iris() #gives a ready made iris dataset from scikit-learn

print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)
X=iris.data
y=iris.target 


X_train, X_test, y_train ,y_test =train_test_split(
  X , y ,
  text_size=0.2,
  random_state=42,
  stratify=y
 )
model = RandomForestClassifier(
 n_estimators=100,
 random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:" , accuracy)
joblib.dump(model, "iris_model.pkl")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
flower_name = iris.target_names[prediction[0]]

print("pridected flower",flower_name)