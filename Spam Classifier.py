#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dataset


# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv('SMSSpamCollection',sep='\t',
                names=['label','message'])


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data['label'].value_counts()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(6,4))
sns.countplot(data['label'],palette= 'Reds')
plt.title("Counting the number of labels",fontsize=15)
plt.xticks(rotation='horizontal')
plt.show()

print(data.label.value_counts())


# In[10]:


# Data cleaning and preprocessing


# In[11]:


import re
import nltk


# In[12]:


# removing stopwords and implementing stemming to get only useful data


# In[13]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[14]:


corpus=[]
for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english') ]
    review=' '.join(review)
    corpus.append(review)


# In[15]:


# Now converting all the sentences into vectors through Countvectorizer and seperating the independent and dependent variable


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()


# In[17]:


y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values


# In[18]:


# spliting the data into tain and test


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[20]:


# using Naive Bayes algorithm to train the data


# In[21]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
spam_detection=nb.fit(X_train,y_train)


# In[22]:


y_pred=spam_detection.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[24]:


cm


# In[25]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


# In[26]:


accuracy


# In[27]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[28]:


# data visualization through heatmap


# In[29]:


sns.heatmap(cm,annot=True)
plt.show()


# In[ ]:




