# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values


#Visualizing the dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
sch=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')           
plt.ylabel('Ecludian Distance')
plt.show() 


#Building Hierachical Clusters
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#visualizing the cluster perfomance
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='category1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='category2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='category3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='yellow',label='category4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='cyan',label='category5')
plt.xlabel('AnnualIncome')
plt.ylabel('spending Score')
plt.legend()
plt.show()





