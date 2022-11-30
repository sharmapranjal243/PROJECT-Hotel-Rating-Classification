#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from pickle import load 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
import rake_nltk
from rake_nltk import Rake
from nltk.tokenize import WhitespaceTokenizer,word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# body of the application 
st.header('HOTEL RATING CLASSIFICATION')
st.markdown('This apllication is build on the machine learning model-Support Vector Machine ')
st.markdown('This application can predict whether the review is positive , negative or neutral')


# In[3]:


# input your review for prediction
input_review=(st.text_area('Type your reviews here....',""""""))


# In[4]:


#loading both svm and tfid Vectorizer intelligence for deployement
logistic=load(open("C:\\Users\\nehas\\logistic_model_deploy.pkl","rb"))
tfid=load(open("C:\\Users\\nehas\\tfid_deploy.pkl","rb"))


# In[5]:


lemmatizer=WordNetLemmatizer()
w_tokenizer=WhitespaceTokenizer()


# In[6]:


stoplist = set(stopwords.words("english"))


# In[7]:


def clean_data(text):
        text=text.lower()
        text=re.sub("\[.*?\]","",text)
        text=re.sub('\S*https?:\S*',"",text)
        text=re.sub("[%s]" % re.escape(string.punctuation),"",text)
        text=re.sub("\w*\d\w*","",text)
        text=re.sub("\n","",text)
        text=re.sub(' +', " ", text)
        return text



clean= clean_data(input_review)


# In[8]:


def lemmatize(txt):
    list_review=[lemmatizer.lemmatize(word=word, pos=tag[0].lower()) 
                 if tag[0].lower() in ['a','r','n','v'] else word for word, tag in pos_tag(w_tokenizer.tokenize(txt))]
    return (' '.join([x for x in list_review if x]))


# In[9]:


#transforming text into numeric
x=tfid.transform([lemmatize(clean)])


# In[10]:


#making prediction
if st.button("Click to make prediction"):
    tfid=tfid.transform([lemmatize(clean)])
    prediction=logistic.predict(tfid)
    prediction =prediction[0]
    if prediction == 'Negative':
        st.error("This is a Negative Review!")
    elif prediction =='Neutral':
        st.warning("This is a Neutral Review!")
    else:
        st.success("This is a Positive Review!")


# In[11]:


#getting keywords using rake module
def get_keywords(text):
    r = Rake(stopwords=set(stoplist), punctuations=set(string.punctuation), include_repeated_phrases=False)
    r.extract_keywords_from_text(input_review)
    words = [re.sub("[%s]" % re.escape(string.punctuation), "", x) for x in r.get_ranked_phrases()]
    words = [x.strip() for x in words if x]
    return words


# In[12]:


result = get_keywords(input_review)


# In[13]:


st.subheader("Influencing Attributes for the Review")


# In[14]:


radio=st.sidebar.radio("Click below to get top Keywords!",("Top 10","Top 20","All"))


# In[15]:


if radio=="Top 10":
    for word in result[:10]:
        st.markdown(word)
elif radio=="Top 20":
    for word in result[:20]:
        st.markdown(word)
else:
    for word in result:
        st.markdown(word)


# In[ ]:





# In[ ]:




