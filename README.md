# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
### Date:
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program

Step 2: Import the required libraries.

Step 3: Upload the csv file and read the dataset.

Step 4: Check for any null values using the isnull() function.

Step 5: From sklearn.tree import DecisionTreeRegressor.

Step 6: Import metrics and calculate the Mean squared error.

Step 7: Apply metrics to the dataset, and predict the output.

Step 8: End the program

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:R suraj pandian
RegisterNumber:  212223080040

import pandas as pd
data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =
train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

*/
```

## Output:
## Position
![91](https://github.com/user-attachments/assets/58e42376-44de-4dd0-8a3b-41303dad377a)

## MSE
![92](https://github.com/user-attachments/assets/b7c4a7c2-8fa8-4671-8cb2-deff9846bff0)

## R2
![93](https://github.com/user-attachments/assets/d21ad0e7-1507-44a7-abb1-73e53c0b1db4)

## Predict
![94](https://github.com/user-attachments/assets/eeb6b7ad-d90d-486c-a198-8f4f2f400c1f)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
