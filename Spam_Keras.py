from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from numpy import random

random.seed(1234)
data=pd.read_csv('Spam.csv',header=0)
d={'ham':1,'spam': 0}
data['Class']=data['Class'].map(d)
X = data['Message']
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

vectorizer=CountVectorizer(max_features=100)
counts_train=vectorizer.fit_transform(x_train)
counts_test=vectorizer.fit_transform(x_test)

model = Sequential()
model.add(Embedding(100, 25))
model.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(counts_train, y_train,batch_size=150,epochs=250,verbose=2,validation_data=(counts_test, y_test))

score, acc = model.evaluate(counts_test, y_test,batch_size=10,verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)
