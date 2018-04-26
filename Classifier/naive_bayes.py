# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')

#Visulaizing the dataset
dataset.head(5)
dataset.tail(5)

#visulaizing the Age and EstimatedSalary columns
dataset['Age'].plot(kind='hist',bins=10)
dataset['EstimatedSalary'].plot(kind='hist',bins=10)

#visualizing the users who bought and not bought the suvs
dataset['Purchased'].plot(kind='hist',bins=3)

#splitting the dataset columns as dependent and independent variable
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#spliting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Building the Classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#prediction
y_pred=classifier.predict(x_test)

#Accuracy
accuracy=classifier.score(x_test,y_test)

#Evaluating the model performance
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Applying k_fold_cross_validation
from sklearn.model_selection import cross_val_score
kfoldaccuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
kfoldaccuracy.mean()
kfoldaccuracy.std()


#visualizing the training set
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Naive_Bayes Training Set')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


#visualizing the testing set
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Naive_Bayes Testing Set')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
















