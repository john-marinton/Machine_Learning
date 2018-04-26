import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#loading the dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#spliting training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#passing the training data to simple linear regression to train the model
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result
y_pred=regressor.predict(x_test)

#checking the predicted value
##print(y_pred)


#visualizing the training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Year of Experiance')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()

#visualizing the test set result
plt.scatter(x_test,y_test)
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title('Salary vs year of experience')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()








