# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing the UCB
import math
d=10
N=10000
ads_selected=[]
number_of_selection=[0]*d
sum_of_rewards=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if (number_of_selection[i] >0):
            average_reward=sum_of_rewards[i]/number_of_selection[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selection[i])
            upper_bound=average_reward+delta_i
        else:
            upper_bound=1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selection[ad]=number_of_selection[ad]+1
    reward=dataset.values[n,ad]
    sum_of_rewards[ad]=sum_of_rewards[ad]+reward
    total_reward=total_reward+reward
             
#Visuailizing the result
plt.hist(ads_selected)
plt.title('Histgram of how many click the partiular ad')
plt.xlabel('different types of Ads')
plt.ylabel('Numbe of times the user clicked')
plt.show()


