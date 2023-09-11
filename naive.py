import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data 
y = iris.target
clf = GaussianNB()
clf.fit(X, y)
y_pred = clf.predict(X)
a=confusion_matrix(y, y_pred)
b=accuracy_score(y, y_pred)
print(a)
print(b)
