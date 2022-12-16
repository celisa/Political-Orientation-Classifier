"""
Description: This script trains a naive bayes model to predict the political affiliation of the user based on their tweet.

"""

# Import libraries

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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim 
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier
from sklearn.metrics import *



# Importing the dataset
train =pd.read_parquet('/Users/pr158admin/Desktop/NLP/Project/NLP_FinalProject/00_source_data/train-00000-of-00001.parquet')
test =pd.read_parquet('/Users/pr158admin/Desktop/NLP/Project/NLP_FinalProject/00_source_data/test-00000-of-00001.parquet')

# Combining the train and test data for cleaning
combine=train.append(test,ignore_index=True)


def remove_pattern(input_text,pattern):
    r= re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text

# Removing twitter handles
combine['tidy_tweet'] = np.vectorize(remove_pattern)(combine['text'],"@[\w]*") 

# All emojis and characters are replaced by white space
combine['tidy_tweet'] = combine['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")

# Removing short words
combine['tidy_tweet'] = combine['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #removing words whose length is less than 3

#stemming
tokenized_tweet = combine['tidy_tweet'].apply(lambda x:x.split()) #it will split all words by whitespace
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #it will stemmatized all words in tweet
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #concat all words into one sentence
combine['tidy_tweet'] = tokenized_tweet

# Naive Bayes Classifier

text = combine['tidy_tweet']
model = CountVectorizer(ngram_range = (2, 2), max_df=0.90 ,min_df=2, stop_words='english')
matrix = model.fit_transform(text).toarray()
df_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
df_output.T.tail(5)

bow_vectorizer = CountVectorizer(ngram_range = (2, 2), max_df=0.90 ,min_df=2 , stop_words='english')
bow = bow_vectorizer.fit_transform(combine['tidy_tweet'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, combine['labels'],
                                                    test_size=0.2, random_state=69)

print("X_train_shape : ",X_train.shape)
print("X_test_shape : ",X_test.shape)
print("y_train_shape : ",y_train.shape)
print("y_test_shape : ",y_test.shape)


model_naive = MultinomialNB().fit(X_train, y_train) 
predicted_naive = model_naive.predict(X_test)

#confusion matrix
plt.figure(dpi=600)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix for Naive Bayes')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig("confusion_matrix.png")
plt.show()


# Accuracy
score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ",score_naive)

# F1 Score
f1_naive = f1_score(predicted_naive, y_test, average='weighted')
print("F1 Score with Naive-bayes: ",f1_naive)

# AUC
auc_naive = roc_auc_score(predicted_naive, y_test)
print("AUC with Naive-bayes: ",auc_naive)

