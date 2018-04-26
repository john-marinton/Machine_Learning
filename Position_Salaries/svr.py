# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#loading the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values

#need not to split the data because need all information

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

#builing the model
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#Visualizing the model
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth of bluff')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()

#prediction
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
print(y_pred)

#visualizing the model in higher resolution and for smoother curve
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len((x_grid)),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth of bluff')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()






