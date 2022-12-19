"""
Description: This script runs some basic EDA for the a naive bayes model.

"""

# Importing libraries

import re #for regular expressions
import nltk #for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split    
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim 
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier
from sklearn.metrics import *
from wordcloud import WordCloud 


# Importing the dataset
train = pd.read_parquet('/Users/kashafali/Documents/Duke/IDS703-NLP/Project/NLP_FinalProject/00_source_data/train-00000-of-00001.parquet')
test = pd.read_parquet('/Users/kashafali/Documents/Duke/IDS703-NLP/Project/NLP_FinalProject/00_source_data/test-00000-of-00001.parquet')

# Combining the train and test data for cleaning
combine=train.append(test,ignore_index=True)


# Cleaning the tweets text by removing certain patterns
def remove_pattern(input_text,pattern):
    r= re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text

def clean_data(tweets_data):
    # Removing twitter handles
    tweets_data['tidy_tweet'] = np.vectorize(remove_pattern)(tweets_data['text'],"@[\w]*") 

    # All emojis and characters are replaced by white space
    tweets_data['tidy_tweet'] = tweets_data['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")

    # Removing short words
    tweets_data['tidy_tweet'] = tweets_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #removing words whose length is less than 3

    #stemming
    tokenized_tweet = tweets_data['tidy_tweet'].apply(lambda x:x.split()) #it will split all words by whitespace
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #it will stemmatized all words in tweet
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #concat all words into one sentence
    tweets_data['tidy_tweet'] = tokenized_tweet
    
    return tweets_data


def word_cloud(tweets_data, label):
    # 0 is Republican, 1 is Democrat
    tweets_data = clean_data(combine)
    normal_words= ' '.join([text for text in tweets_data['tidy_tweet'][tweets_data['labels']==label]])
    wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    #plt.title('Word Cloud for ' + label)
    if label == 0:
        plt.savefig("Republican_wordcloud.png")
    else:
       plt.savefig("Democrat_wordcloud.png")
    #plt.savefig(label + "_wordcloud.png")
    return plt.show()

# Word Cloud for Republican
word_cloud(combine, 0)

# Word Cloud for Democrat
word_cloud(combine, 1)

