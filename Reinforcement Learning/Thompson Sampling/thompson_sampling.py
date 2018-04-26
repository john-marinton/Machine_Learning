# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 12:16:37 2018

@author: John Marinton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thomson Sampling
import random
d=10
N=10000
ads_selected=[]
number_of_times_reward_1=[0]*d
number_of_times_reward_0=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_times_reward_1[i]+1,
                                       number_of_times_reward_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)        
    reward=dataset.values[n,ad]
    if reward==1:
        number_of_times_reward_1[ad]=number_of_times_reward_1[ad]+1
    else:
        number_of_times_reward_0[ad]=number_of_times_reward_0[ad]+1
    total_reward=total_reward+reward 


#visualizing the result
plt.hist(ads_selected)
plt.title('Thompson Sampling')
plt.xlabel('Types Of Ads')
plt.ylabel('Number of times it has been selected')
plt.show()       
        
        
        
        
        
        
        
