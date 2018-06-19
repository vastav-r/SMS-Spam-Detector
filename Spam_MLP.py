import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('Spam.csv',header=0)
d={'ham':1,'spam': 0}
data['Class']=data['Class'].map(d)
X = data['Message']
y = data['Class']

vectorizer=CountVectorizer(max_features=100)
counts_train=vectorizer.fit_transform(X)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(counts_train,y)
MLPClassifier(activation='relu', alpha=1e-03, batch_size='auto',
       beta_1=0.5, beta_2=0.3,
       epsilon=1e-05, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.003, max_iter=2000000,random_state=1)
print("Enter")
word=input()
x_test=[word]
counts_test=vectorizer.transform(x_test)
y_pred=clf.predict(counts_test)
print(y_pred)

