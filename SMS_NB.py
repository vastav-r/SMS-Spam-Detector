import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



def spam():
    print("Enter the Message :")
    word=input()
    example=[word]
    example_count=vectorizer.transform(example)

    predictions=clf.predict(example_count)
    print("\nYour message "+str(example)+" is classified as "+str(predictions)+"\n")



            
data=pd.read_csv('Spam.csv',header=0)

vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(data['Message'].values)

clf=MultinomialNB()

targets=data['Class'].values
clf.fit(counts,targets)

while(1):
    print("1: Try the SPAM classifier, q: exit") 
    ch=input()
    if(ch!='q'):
        spam()
    else:
        exit()

