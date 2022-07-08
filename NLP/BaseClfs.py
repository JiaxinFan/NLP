#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import pandas as pd
import argparse
import sys
import nltk
import json
from pathlib import Path
from sklearn import svm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics


# In[115]:


def convert_label(label):
    if label == 'nonrumour':
        return 0
    else:
        return 1


# In[116]:


def processText(datalist, tag):
    if tag != 'test':
        datalist = sorted(datalist, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
    else:
        datalist = sorted(datalist,
                          key=lambda x: time.mktime(time.strptime(x["created_at"], '%a %b %d %H:%M:%S +0000 %Y')))
    processed_text = ""
    for item in datalist:
        new_text = []
        for text in item["text"].split(" "):
            text = text.replace('\n', '').replace('\r', '').lower()
            if text.startswith('@') and len(text) > 1:
                text = '@user'
            if text.startswith('http'):
                text = 'http'
            new_text.append(text)
        processed = " ".join(new_text)
        processed_text = processed_text + processed
    return processed_text


# In[117]:


def processData(datafile,labelfile):
    idList=[]
    tweetlist=[]
    label=[]
    ###for return
    tweets=dict()
    ######
    with open(datafile, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  
            line=line.strip().split(",")
            idList.append(line)
    with open(labelfile, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  
            label.append(line)
    ######
    for i in range(len(idList)):
        templist=[]
        for j in range(len(idList[i])):
            jsonfile="./train/" + idList[i][j] + ".json"
            if (Path(jsonfile)).is_file():
                templist.append(json.load(open(jsonfile, "r")))
        tweetlist.append(templist)
    ####
    tweet=[]
    for i in range(len(idList)):
        tweet.append(processText(tweetlist[i],"train"))
    tweets["text"]=tweet
    tweets["label"]=label
    ####
    return tweets


# In[181]:


def BaselineClf(OutputState):
    ##preprocess
    tweets_train=processData("train.data.txt","train.label.txt")
    tweets_dev=processData("dev.data.txt","dev.label.txt")
    tweets_test=processData("test.data.txt","dev.label.txt")

    tfidf = TfidfVectorizer(max_features=None,strip_accents='unicode',use_idf=1,smooth_idf=2,stop_words=stopwords1,sublinear_tf = True,ngram_range=(1,3))
    train_feature = tfidf.fit_transform(tweets_train['text']) 
    dev_feature=tfidf.transform(tweets_dev['text'])
    test_feature=tfidf.transform(tweets_test['text'])
    ##baseline
    baselabel=[]
    with open("dev.baseline.txt", "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  
                baselabel.append(line)
    print("dev.baseline.txt_acc:%f"%accuracy_score(baselabel ,tweets_dev["label"]))
    print("")
  
    ##baseline clfs
    clf = svm.SVC(C=1.0, kernel='linear', degree=3,gamma='auto',decision_function_shape='ovr' )
    clf.fit(train_feature, tweets_train['label'])
    svmpred = clf.predict(dev_feature)
    print("SVM_acc:%f"%metrics.precision_score(tweets_dev["label"], svmpred, average='macro'))
    print("SVM_recall:%f"%metrics.recall_score(tweets_dev["label"], svmpred, average='micro'))
    print("SVM_f1:%f"%metrics.f1_score(tweets_dev["label"], svmpred, average='weighted'))
    print("")
   
    
    lr = LogisticRegression(max_iter=2000,penalty="l1",solver='saga')
    lr.fit(train_feature, tweets_train['label'])
    pred=lr.predict(dev_feature)
    print("LogisticRegression_acc:%f"%metrics.precision_score(tweets_dev["label"], pred, average='micro'))
    print("LogisticRegression_recall:%f"%metrics.recall_score(tweets_dev["label"], pred, average='micro'))
    print("LogisticRegression_f1:%f"%metrics.f1_score(tweets_dev["label"], pred, average='weighted'))
    print("")

    
    Nbclf = MultinomialNB(alpha=0.1)
    Nbclf.fit(train_feature, tweets_train["label"])
    pred = Nbclf.predict(dev_feature)
    MultinomialNB_acc=accuracy_score(pred ,tweets_dev["label"])
    
    print("MultinomialNB_acc:%f"%metrics.precision_score(tweets_dev["label"], pred, average='micro'))
    print("MultinomialNB_recall:%f"%metrics.recall_score(tweets_dev["label"], pred, average='micro'))
    print("MultinomialNB_f1:%f"%metrics.f1_score(tweets_dev["label"], pred, average='weighted'))
    
    if OutputState==True:
        pred_test = clf.predict(test_feature)
        svmresult=[]
        for i in range(len(pred_test)):
            temp=[]
            temp.append(i)
            temp.append(convert_label(pred_test[i]))
            svmresult.append(temp)

        column=['Id','Predicted'] 
        test=pd.DataFrame(columns=column,data=svmresult)
        test.to_csv('./testresult.csv',index=False) 
        


# In[182]:


## enter True if need a CSV file
BaselineClf(OutputState=False)

