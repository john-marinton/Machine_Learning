# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Visualizing the dataset
dataset.iloc[:,[2,3]].count().plot(kind='bar')


#Describing
dataset.describe()

#head and tail of dataset
dataset.head(5)
dataset.tail(5)

#spliting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Building the svm model
from sklearn.svm import SVC
#classifier=SVC(kernel='linear',random_state=0)      #accuracy=90
#classifier.fit(x_train,y_train)

#changing the kernel to rbf accuracy Checking
classifier=SVC(kernel='rbf',random_state=0)           #accuracy=93
classifier.fit(x_train,y_train)

#changing the kernel to poly accuracy Checking
#classifier=SVC(kernel='poly',degree=3,random_state=0)   #accuracy=86
#classifier.fit(x_train,y_train)

#changing the kernel to poly accuracy Checking
#classifier=SVC(kernel='sigmoid',random_state=0)   #accuracy=74 #overfitted
#classifier.fit(x_train,y_train)



#Prediction
y_pred=classifier.predict(x_test)

#Evaluating the model usind confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Accuracy
accuracy=classifier.score(x_test,y_test)

#Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score
kfoldaccuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
kfoldaccuracy.mean()
kfoldaccuracy.std()

#implementing the grid search to find the best model and best hyperparameters
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
             {'C':[1,10,100,1000],'kernel':['rbf'],
              'gamma':[0.5,0.1,0.2,0.4,0.3,0.6,0.7]},
             {'C':[1,10,100,1000],'kernel':['poly'],'degree':[2,3,4],
              'gamma':[0.5,0.1,0.2]}]

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,
                         scoring='accuracy',n_jobs=-1,cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_model=grid_search.best_params_






#Visualizing the classifier training set
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)) 
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('SVM Training Set')
plt.xlabel('Ages')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing the classifier Testing set
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)) 
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('SVM Test Set')
plt.xlabel('Ages')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
