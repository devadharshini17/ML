from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
# Load the data
iris = datasets.load_iris()

X = iris.data 
y = iris.target
# Create a Logistic Regression classifier
clf = LogisticRegression()
# Train the classifier
clf.fit(X, y)
# Make predictions
y_pred = clf.predict(X)
# Create a confusion matrix
confusion_matrix(y, y_pred)
# Calculate the accuracy
accuracy_score(y, y_pred)
