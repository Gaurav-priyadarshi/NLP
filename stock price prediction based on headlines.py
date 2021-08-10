# kaggle competition stock price predicton based on news headlines
import pandas as pd
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv(r'C:/Users/gaurav/Desktop/python download video/pythoncode/ML code/stock price healines nlp/Combined_News_DJIA.csv')
null_value = df.isnull().sum()   
sns.countplot(x = df['Label'])

train_data = df[df['Date']<'20150101']
test_data = df[df['Date']>'20141231']
test_data.reset_index(inplace=True,drop=True)


x_train = train_data.iloc[:,2:27]
x_test = test_data.iloc[:,2:27]
y_train = train_data.iloc[:,1]
y_test = test_data.iloc[:,1]

#cleaning the text of train and test
columns = [i for i in x_test.columns]

#cleaning the data

def clean_data(dataset):
    dataset.replace('[^a-zA-Z]',' ',regex=True,inplace = True)
    return dataset 

# converting all the cloumn data into a single column

def combine_data(data):
    headlines = []
    for i in range(len(data)):
        headlines.append(' '.join(str(x)  for x in data.iloc[i,:]))
    return headlines
        
# stemming and lemmatization of word
def data_stem(data,lemmatization):
    stem_dataset = []
    for i in range(len(data)):
        review = data[i].lower()
        review = review.split()
        review = [lemmatization.lemmatize(word)  for word in review  if word not in set(stopwords.words('english'))]
        stem_dataset.append(' '.join(review))
    return(stem_dataset)

#converting the test data into vector form

def vector_data(data,model):
    vector_dataset = model.fit_transform(data)
    return vector_dataset

# applying all function in train and test datasets

#clening data
x_clean_train  = clean_data(x_train)
x_clean_test = clean_data(x_test)

#combining the data
x_train_combine = combine_data(x_clean_train)
x_test_combine  = combine_data(x_clean_test)

# stemming the data
from nltk.stem import WordNetLemmatizer
lemmatization = WordNetLemmatizer()
x_train_stem = data_stem(x_train_combine,lemmatization)
x_test_stem = data_stem(x_test_combine,lemmatization)

#vector data
from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer()
x_train_vector = vector_data(x_train_stem,model)
x_test_vector = model.transform(x_test_stem)
x_train_vector = x_train_vector.toarray()
x_test_vector = x_test_vector.toarray()

# traing our model
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=200,criterion='entropy')
random_forest.fit(x_train_vector,y_train)

#prediction

y_pred = random_forest.predict(x_test_vector)

#confusion matrix and score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
roc_score = roc_auc_score(y_test,y_pred)

#roc_auc_curve
fpr,tpr,thresholds = roc_curve(y_test,y_pred)

#plotting roc_auc_curve
import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color = 'red',label = 'roc',linestyle='dotted')
plt.plot([0,1],[0,1],color = 'blue',linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('roc curve')
plt.legend()
plt.show()






        


        




            
            
            




