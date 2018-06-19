import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import random


def spam():
    
    
    example_count=vectorizer.transform(X_test)

    y_pred=clf.predict(example_count)
    print(y_pred)
    print(accuracy_score(y_test,y_pred)*100)
   # print("\nYour message "+str(example)+" is classified as "+str(predictions)+"\n")




data=pd.read_csv('Spam.csv',header=0)

X = data['Message']
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y)


vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(X_train)

clf=MultinomialNB()

clf.fit(counts,y_train)
spam()

from sklearn.model_selection import cross_val_score
cv=cross_val_score(clf,counts,y,cv=10)
cv.mean()
