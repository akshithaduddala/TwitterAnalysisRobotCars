import pandas as pd
import re
import urllib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


data = pd.read_csv('TestData.csv')


data.dropna(subset=['text'], inplace=True)
#data.dropna(subset=['emotion'], inplace=True)
data=data[data.index%2==0]


data["text"]=data.text.str.replace(r'^b','') 
data["text"]=data.text.str.replace(r'https?:\/\/.*\/[a-zA-Z0-9]*', '') 
data["text"]=data.text.str.replace(r'@[a-zA-Z0-9]{1,10}', '') 
data["text"]=data.text.str.replace(r'\$[a-zA-Z0-9]*', '')
data["text"]=data.text.str.replace(r'[0-9]*','')
data["text"]=data.text.str.replace(r'\\[a-z A-Z]{1,2}','')
data["text"]=data.text.str.replace(r'\:','')
data["text"]=data.text.str.replace(r'\\n','')
data["text"]=data.text.str.replace(r'\#','')
data["text"]=data.text.str.replace(r'\/','')
data["text"]=data.text.str.replace(r'\'','')
data["text"]=data.text.str.replace(r'\"','')
data["text"]=data.text.str.replace(r'\-','')
data["text"]=data.text.str.replace(r'\?','')
data["text"]=data.text.str.replace(r'\_','')
data["text"]=data.text.str.replace(r'%','')
data["text"]=data.text.str.replace(r'\,','')
data["text"]=data.text.str.replace(r'.','')
data["text"]=data.text.str.replace(r'\&amp','')
data["text"]=data.text.str.replace(r';','')
data["text"]=data.text.str.replace(r'!','')
data["text"]=data.text.str.replace(r'\\s','')
data["text"]=data.text.str.replace(r'\)','')
data["text"]=data.text.str.replace(r'\(','')
data["text"]=data.text.str.replace(r'\+','')
data["text"]=data.text.str.replace(r'\=','')


vectorizer = TfidfVectorizer(min_df=2,max_df=0.9,lowercase="True",stop_words="english")


X = vectorizer.fit_transform(data.text.values.astype('U'))
y=data.emotion
data['X']=list(X)
data['y']=data.emotion


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X[0:875], y[0:875])


result=[] 
result=clf.predict(X[0:6966])
val=np.arange(0,13931,2)
series=pd.Series(result,index=val)
#print(series)

data["NaiveBayes"]=series

from sklearn.metrics import accuracy_score
data.head()
data.index
accuracy_score(series[0:875],y[0:875])

from sklearn.metrics import accuracy_score
accuracy_score(series[876:1077],y[876:1077])
#print(series[877:1077])


from sklearn import svm
clf_svm = svm.SVC(kernel="linear", verbose=3)
clf_svm.fit(X[0:875], y[0:875])


res=[]
res=clf_svm.predict(X[0:6966])
lis=np.arange(0,13931,2)
ser=pd.Series(res,index=lis)
#print(ser)
data['SVM']=ser


accuracy_score(ser[0:875],y[0:875])


accuracy_score(ser[876:1077],y[876:1077])


data.to_csv('Result.csv')


get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.pyplot as plt

#Naive Bayes on Train Data
from collections import Counter
c=Counter(y[:875])
s=Counter(series[:875])
print("\n")
print("Actual Train Data Sentiment Count:", c)
print("Naive Bayes Train Data Sentiment Count:", s)
n_groups=3
yemo=(463,358,54)
NBemo=(542,333,0)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 1
rects1 = plt.bar(index, yemo, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Actual sentiment of Train Data')
rects2 = plt.bar(index + bar_width, NBemo, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Naive Bayes predicted sentiment of Train Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(index + bar_width, ('Positive', 'Neutral', 'Negative'))
plt.legend()
plt.title('Accuracy of Naive Bayes Model- Train Data')
plt.tight_layout()
plt.show()

#Naive Bayes on Test Data
from collections import Counter
ct=Counter(y[877:1078])
st=Counter(series[876:1077])
print("Actual Test Data Sentiment Count:", ct)
print("Naive Bayes Test Data Sentiment Count:", st)
n_groups=3
ytestemo=(150,24,27)
NBtestemo=(151,50,0)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 1
rects1 = plt.bar(index, ytestemo, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Actual sentiment of Test Data')
rects2 = plt.bar(index + bar_width, NBtestemo,bar_width,
                 alpha=opacity,
                 color='r',
                 label='Naive Bayes predicted sentiment of Test Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(index + bar_width, ('Positive', 'Neutral', 'Negative'))
plt.legend()
plt.title('Accuracy of Naive Bayes Model- Test Data')
plt.tight_layout()
plt.show()

#SVM on Train data
from collections import Counter
cs=Counter(y[:875])
ss=Counter(ser[:875])
print("Actual Test Data Sentiment Count:", cs)
print("SVM Test Data Sentiment Count:", ss)
n_groups=3
yemo=(463,358,54)
SVMemo=(462,377,36)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8
rects1 = plt.bar(index, yemo, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Actual sentiment of Train Data')
rects2 = plt.bar(index + bar_width, SVMemo,bar_width,
                 alpha=opacity,
                 color='r',
                 label='SVM predicted sentiment of Train data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(index + bar_width, ('Positive', 'Neutral', 'Negative'))
plt.legend()
plt.title('Accuracy of SVM Model- Train Data')
plt.tight_layout()
plt.show()

#SVM on Test Data
from collections import Counter
ct=Counter(y[876:1078])
st=Counter(ser[876:1077])
print("Actual Test Data Sentiment Count:", ct)
print("SVM Test Data Sentiment Count:", st)
n_groups=3
ytestemo=(150,24,28)
SVMtestemo=(141,60,0)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8
rects1 = plt.bar(index, ytestemo, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Actual sentiment of Test Data')
rects2 = plt.bar(index + bar_width, SVMtestemo,bar_width,
                 alpha=opacity,
                 color='r',
                 label='SVM predicted sentiment of Test Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(index + bar_width, ('Positive', 'Neutral', 'Negative'))
plt.legend()
plt.title('Accuracy of SVM Model- Test Data')
plt.tight_layout()
plt.show()


val=(data.loc[data['emotion'] == 'negative'])

v=(data.loc[data['SVM'] == 'negative'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
wordfreq={}
for tweet in val["text"]:
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            if(w not in wordfreq):
                wordfreq[w]=0
            wordfreq[w]+=1

from operator import itemgetter
sorted(wordfreq.items(), key=itemgetter(1))

