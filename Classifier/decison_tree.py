# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#Loading the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')

#Visualizing the dataset
dataset['Age'].plot(kind='hist',bins=10)
dataset['EstimatedSalary'].plot(kind='hist',bins=10)

#splitting the dataset into dependent and independent varible
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Spliting into training and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Building the classifier model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#Prediction
y_pred=classifier.predict(x_test)

#Accuracy
accuracy=classifier.score(x_test,y_test)

#Evaluating the classifier model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#visualizing the decison tree classifier(Training Set)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import matplotlib.style as style

#style.use('ggplot')

x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,
             cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())                  
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('decison tree classifier(Training Set)')    
plt.legend()
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')

#visualizing the decison tree classifier(Testing Set)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import matplotlib.style as style

#style.use('ggplot')

x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,
             cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())                  
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('decison tree classifier(Testing Set)')    
plt.legend()
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')








