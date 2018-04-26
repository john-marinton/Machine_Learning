# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#loading the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Describing
dataset.describe()

#visualising the dataset
dataset.iloc[:,[2,3]].count().plot(kind='bar')
dataset.iloc[:,4].plot()

#Splitting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#Building the model
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)        #Accuracy=93
classifier.fit(x_train,y_train)

#Kernel Poly
#from sklearn.svm import SVC
#classifier=SVC(kernel='poly',random_state=0,degree=3)        #Accuracy=86
#classifier.fit(x_train,y_train)

#kernel sigmoid
#from sklearn.svm import SVC
#classifier=SVC(kernel='sigmoid',random_state=0)  #Accuracy=74 not a best model
#classifier.fit(x_train,y_train)                    #underfitted


#prediction
y_pred=classifier.predict(x_test)

#accuracy
accuracy=classifier.score(x_test,y_test)

#Evaluating the model pereformance
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Visualizing the Training Set model
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Svm Training Set')
plt.xlabel('Ages')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    

#Visualizing the Testing Set model
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Svm Training Set')
plt.xlabel('Ages')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   

