# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#loading the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values

#building the model
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#Accuracy
accuracy=regressor.score(x,y)

#prediction
y_pred=regressor.predict(6.5)

#visualizing the decison tree plot         """This plot is not correct"""
#import matplotlib.pyplot as plt
#plt.scatter(x,y,color='blue')
#plt.plot(x,regressor.predict(x),color='red')
#plt.title('Truth or bluff')
#plt.xlabel('Level')
#plt.ylabel('Salaries')
#plt.show()

#visualizing the decison tree with higher resolution and smoother curve
import matplotlib.pyplot as plt
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()
