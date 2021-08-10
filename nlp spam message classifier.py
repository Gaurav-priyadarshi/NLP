import pandas as pd
messages  = pd.read_csv('C:/Users/gaurav/Desktop/python download video/pythoncode/ML code/smsspamcollection/SMSSpamCollection', sep='\t',names=['label','message'])

#data cleaning and processing
import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
ps = PorterStemmer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)  for word in review  if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#crearing bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['label'],drop_first=True).values 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=0)

#training the model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model  = MultinomialNB()
spam_detect_model.fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

#len(messages)
