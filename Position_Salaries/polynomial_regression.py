# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading Dataseet
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#can't split training and testing coz the model needs all information


#linear regression
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(x,y)

#Score accuracy
#accuracy1=lreg.score(x,y)
#print(accuracy1)


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
preg=PolynomialFeatures(degree=4)
x_poly=preg.fit_transform(x)
lreg_2=LinearRegression()
lreg_2.fit(x_poly,y)

#Score accuracy
#accuracy2=lreg_2.score(x_poly,y)
#print(accuracy2)

#visualizing the linear Regression
plt.scatter(x,y,color='red')
plt.plot(x,lreg.predict(x),color='blue')
plt.title('Finiding the bluff or not')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#Using np.arrange to create a higher resolution with a smoother curve
#by incementing each level with 0.1 so it 1 to 9.9 values
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid)),1)
#visualizing the polynomialRegression
plt.scatter(x,y,color='red')
plt.plot(x_grid,lreg_2.predict(preg.fit_transform(x_grid)),color='blue')
plt.title('Finiding the bluff or not')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting the specified employee salaries using his level in linear regression
lreg.predict(6.5)

#Predicting the specified employee salaries using his level in polynomial regression
lreg_2.predict(preg.fit_transform(6.5))
