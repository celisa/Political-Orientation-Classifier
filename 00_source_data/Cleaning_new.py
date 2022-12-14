import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import warnings
import nltk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import Word
import os

nltk.download('punkt')
nltk.download('wordnet')

#read data 
train_data =pd.read_parquet('./train-00000-of-00001.parquet')
test_data =pd.read_parquet('./test-00000-of-00001.parquet')


def clean_data(tweets_df, remove_stopwords=False):
    """Clean the data by removing URLs, converting to lowercase and removing @s and #s from the tweet"""
    tweets_df['text_new']=tweets_df['text'].astype('str')
    warnings.filterwarnings("ignore")

    #replace nan with empty string
    tweets_df["text_new"] = tweets_df["text_new"].fillna('')

    # remove all URLs from the text
    tweets_df["text_new"] = tweets_df["text_new"].str.replace(r"http\S+", "")

    # remove all mentions from the text and replace with generic flag
    tweets_df["text_new"] = tweets_df["text_new"].str.replace(r"@\S+", "")

    # remove all hashtags from the text
    tweets_df["text_new"] = tweets_df["text_new"].str.replace(r"#", "")

    # lowercase all text
    tweets_df["text_new"] = tweets_df["text_new"].str.lower()
    
    #remove punctuations, numbers and words with length of two characters
    string1 = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    def remove_punct(text):
        text  = "".join([char for char in text if char not in string1])
        text = re.sub('[0-9]+', '', text)
        text= re.sub(r'\b\w{1,2}\b', '', text)
        return text
    
    tweets_df["text_new"] = tweets_df["text_new"].apply(lambda x: remove_punct(x))

    # removing word lengths less than 3 
    #tweets_df.text_new.str.replace(r'\b(\w{1,3})\b', '')

    #tweets_df.text_new.apply(lambda txt: ''.join(TextBlob(txt).correct()))

    if remove_stopwords:
        # remove stopwords
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        tweets_df["text_new"] = tweets_df["text_new"].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )

    
    # checking for uncommon words
    fdist = FreqDist(tweets_df["text_new"])
    tweets_df["uncommon"] = tweets_df["text_new"].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))
    assert (tweets_df["uncommon"]=="").all()
    #tokenisation
    tweets_df["text_new"].apply(word_tokenize)
    tweets_df["text_new"] = tweets_df["text_new"].str.split()

    return tweets_df



cleaned_train = clean_data(train_data,remove_stopwords=True)
cleaned_test = clean_data(test_data,remove_stopwords=True)

cleaned_train.to_csv('cleaned_train.csv',index=False)
cleaned_test.to_csv('cleaned_test.csv',index=False)

cleaned_train = pd.read_csv('cleaned_train.csv')
cleaned_test = pd.read_csv('cleaned_test.csv')

# convert the 'text_new' field to a list of words - currently in str
cleaned_test['text_new'] = cleaned_test['text_new'].apply(lambda x: x.strip('][').replace("'",'').split(', '))
cleaned_train['text_new'] = cleaned_train['text_new'].apply(lambda x: x.strip('][').replace("'",'').split(', '))

# create a dictionary of all misspelled words and their corrected spelling
all_words = np.array(cleaned_train['text_new']).tolist() + np.array(cleaned_test['text_new']).tolist()

all_words_flat = [x for row in all_words for x in row]
unique_words = set(all_words_flat)
print(f"There are {len(unique_words)} unique words in the dataset.")

def create_misspelled_dict(unique_words):
    """create a dictionary for all misspelled words and their corrected spelling"""
    misspelled_dict = {}
    for word in unique_words:
        corr_word = Word(word)
        result = corr_word.spellcheck()
        if word != result[0][0]:
            misspelled_dict[word] = result[0][0]
    return misspelled_dict

#check if file exists
if os.path.exists('misspelled_dict.csv'):
    misspelled_dict = pd.read_csv('misspelled_dict.csv',index_col=0).to_dict()
else:
    misspelled_dict = create_misspelled_dict(unique_words)
    pd.DataFrame.from_dict(misspelled_dict, orient='index').to_csv('misspelled_dict.csv')

cleaned_train['text_new'] = cleaned_train['text_new'].apply(lambda x: [word if word not in misspelled_dict.keys() else misspelled_dict[word] for word in x])
cleaned_test['text_new'] = cleaned_test['text_new'].apply(lambda x: [word if word not in misspelled_dict.keys() else misspelled_dict[word] for word in x])


#stemming 


ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

cleaned_train["text_new"] = cleaned_train["text_new"].apply(lambda x: stemming(x))

cleaned_test["text_new"] = cleaned_test["text_new"].apply(lambda x: stemming(x))



#lemmetisation 

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text
cleaned_train['text_new'] = cleaned_train['text_new'].apply(lambda x: lemmatizer(x))
cleaned_test['text_new'] = cleaned_test['text_new'].apply(lambda x: lemmatizer(x))

cleaned_train=cleaned_train.drop(['date','id','username','text','uncommon'], axis=1)
cleaned_test=cleaned_test.drop(['date','id','username','text','uncommon'], axis=1)

cleaned_train.to_csv('cleaned_train.csv', index=False)
cleaned_test.to_csv('cleaned_test.csv', index=False)

print("Data Processing Done!")
