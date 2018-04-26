# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#CLeaning the dataset
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

#Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)   
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#spliting into train and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


def model_accuracy(cm):
    
    #splitting into true positive,true negative,false positive,false negative
    TP=cm[1][1]
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    
    accuracy=(TP+TN)/(TP+TN+FN+FP)
    print('Accuracy',accuracy)
    
    precision=TP/(TP+FP)
    print('Precision',precision)
    
    recall=TP/(TP+FN)
    print('Recall',recall)
    
    f1score=2*precision*recall/(precision+recall)
    print('F1 Score',f1score)

def naive_bayes(x_train,x_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier=GaussianNB()
    classifier.fit(x_train,y_train)
    
    #prediction
    y_pred=classifier.predict(x_test)
    
    #confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    
    print('Naive Bayes Model Performance ')
    model_accuracy(cm)
 

    
def logistic_regression(x_train,x_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    
    #prediction
    y_pred=classifier.predict(x_test)
    
    #confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    
    print('Logistic Regression Model Performance ')
    model_accuracy(cm)

def random_forest(x_train,x_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=30,random_state=0)
    classifier.fit(x_train,y_train)
    
    #prediction
    y_pred=classifier.predict(x_test)
    
    #confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    
    print('Random Forest Model Performance ')
    model_accuracy(cm)

def decision_tree(x_train,x_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(random_state=0)
    classifier.fit(x_train,y_train)
    
    #prediction
    y_pred=classifier.predict(x_test)
    
    #confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    
    print('Decision Tree Model Performance ')
    model_accuracy(cm)
    
def svm(x_train,x_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier=SVC(kernel='rbf',random_state=0)
    classifier.fit(x_train,y_train)
    
    #prediction
    y_pred=classifier.predict(x_test)
    
    #confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    
    print('SVM with RBF kernel Model Performance ')
    model_accuracy(cm)


naive_bayes(x_train,x_test,y_train,y_test)
logistic_regression(x_train,x_test,y_train,y_test)
random_forest(x_train,x_test,y_train,y_test)
decision_tree(x_train,x_test,y_train,y_test)




