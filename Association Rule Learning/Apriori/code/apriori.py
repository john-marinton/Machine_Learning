# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#loading the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)

#importing the dataset
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#building the model    
from apyori import apriori    
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visualizing the rules
result=list(rules)

print(result,sep=',')