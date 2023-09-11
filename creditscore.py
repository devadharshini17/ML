import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv('credit_score_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('credit_score', axis=1), data['credit_score'], test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
age = int(input("Enter your age: "))
income = int(input("Enter your income: "))
education = input("Enter your education level : ")
marital_status = input("Enter your marital status : ")
credit_score = model.predict([[age, income, education, marital_status]])
print("Your predicted credit score is:", credit_score)
