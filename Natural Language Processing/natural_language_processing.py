# -*- coding: utf-8 -*-

#Natural Language Processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)    
    corpus.append(review)



#Building bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()    
y=dataset.iloc[:,1].values

#splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

    
#building Naivebayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#prediction
y_pred=classifier.predict(x_test)

#confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Assinging true positive,true negative and false positive and false negative
TN=cm[0][0]
TP=cm[1][1]
FP=cm[0][1]
FN=cm[1][0]


#Acurracy
accuracy=(TP+TN)/(TP+TN+FP+FN)

#Precision
Precision = TP / (TP + FP)

#Recall
Recall = TP / (TP + FN)

#F1 Score
F1_Score = 2 * Precision * Recall / (Precision + Recall)

#kfold corss validation
from sklearn.model_selection import cross_val_score
kfold=cross_val_score(estimator=classifier,X=x_train,y=y_train,scoring='accuracy',
                      cv=10)
kfold.mean()
kfold.std()


