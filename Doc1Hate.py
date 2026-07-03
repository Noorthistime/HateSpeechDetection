#Hate speech detection [Code]

#Import Dataset from kaggle.com [It would be labeled_data.csv file weighing 2.55 mb]
#"Here is the Dataset link -- https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset "
#Importing Libraries

import pandas as pd 
import numpy as np
import sklearn

dataset = pd.read_csv("labeled_data.csv")
dataset
dataset.isnull()
dataset.isnull().sum()
dataset.info()
dataset.describe()

dataset["labels"] = dataset["class"].map({
    0: "Hate Speech", 
    1: "Offensive language", 
    2: "No hate or offensive language"
})

data = dataset[["tweet" , "labels"]].copy()
data

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

stopwords = set(stopwords.words("english"))

#Import Stemming
stemmer = nltk.SnowballStemmer("english")

#Data cleaning
def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopwords]
    text = " ".join(text)
    #Stemming the text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text) 
    return text 

data["tweet"] = data["tweet"].apply(clean_data)
data

#Creating a NumPy Array
x =  np.array(data["tweet"])
y =  np.array(data["labels"])

x 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
x = cv.fit_transform(x)
x

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
x_train

#Building ML-Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

import seaborn as sns
import matplotlib.pyplot as ply
# %matplotlib inline

sns.heatmap(cm, annot = True, fmt="f", cmap = "YlGnBu")

from sklearn.metrics import accuracy_score 
accuracy_score(y_test, y_pred)

sample = "Let's unite and kill all the people who are protesting against the government"
sample = clean_data(sample)
data1 = cv.transform([sample]).toarray()
data1
dt.predict(data1)