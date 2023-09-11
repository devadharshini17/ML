import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
# Load the data
iris = load_iris()
data = pd.DataFrame(iris.data, columns = ['sepal_length','sepal_width','petal_length','petal_width'])
data['species'] = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('species', axis=1), data['species'], test_size=0.25, random_state=0)
# Create a KNN model
model = KNeighborsClassifier()
# Fit the model to the training data
model.fit(X_train, y_train)
# Get user input
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
# Predict the species
species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
# Print the predicted species
print("The predicted species is:", species)
