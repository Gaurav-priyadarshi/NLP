# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:36:04 2021

@author: gaurav
"""

#fake news classifier

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer

train_data = pd.read_csv(r'C:/Users/gaurav/Desktop/python download video/python data/fake news classifier/train.csv/train.csv')
test_data = pd.read_csv(r'C:/Users/gaurav/Desktop/python download video/python data/fake news classifier/test.csv/test.csv')

#checking nan value in train and test data
train_nan = train_data.isnull().sum()
test_nan = test_data.isnull().sum()

# dropping the nan value  
train_data = train_data.dropna()
test_data = test_data.dropna()

# re indexing of the dataset
train_data.reset_index(inplace = True,drop = True)
test_data.reset_index(inplace = True,drop = True)

# creating dependent and independent feature

X_train = train_data.iloc[:,1:4]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,1:4]

# cleaning the data
def clean_data(dataset):
    dataset.replace('[^a-zA-Z]',' ',regex =True,inplace=True )
    return dataset

# stemming of the text

def stem_data(data,stem):
    news = []
    for i in range(len(data)):
        review = data.iloc[i].lower()
        review = review.split()
        review = [stem.lemmatize(word) for word in review  if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        news.append(review)
    return news
    
#converting the test data into vector form

def vector_data(data,model):
    X_train_final = model.fit_transform(data).toarray()
    return X_train_final

# applying all the function in training and test dataset

#clean dataset

X_train_clean = clean_data(X_train['title'])
X_test_clean = clean_data(X_test['title'])

#stemming the data

stem  =WordNetLemmatizer()

X_train_stem = stem_data(X_train_clean,stem) 
X_test_stem = stem_data(X_test_clean,stem)

# converting into vector form

model = CountVectorizer(max_features=5000)

X_final_train = vector_data(X_train_stem,model)
X_final_test = model.transform(X_test_stem).toarray()

#training our model

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

classifier1 = RandomForestClassifier(n_estimators=200,criterion='entropy')
classifier2 = MultinomialNB()

def training_model(X,y,test,classifier):
    fake_detect_model = classifier.fit(X,y)
    y_pred  = fake_detect_model.predict(test)
    return y_pred

prediction1 = training_model(X_final_train,y_train,X_final_test,classifier1)
prediction2 = training_model(X_final_train,y_train,X_final_test,classifier2)


















