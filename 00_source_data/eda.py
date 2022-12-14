import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
#read data 
train_data =pd.read_parquet('./train-00000-of-00001.parquet')
test_data =pd.read_parquet('./test-00000-of-00001.parquet')

#drop empty rows with no tweets
train_data.drop(train_data[train_data['text'].str.len()==0].index, inplace = True)
test_data.drop(test_data[test_data['text'].str.len()==0].index, inplace = True)
#check if all labels are classified as 0 and 1 
assert (train_data["labels"]==0|1).all
#summary statistics

#shape of train data 
print("Shape of train dataset is "+str(train_data.shape))
print("Shape of test dataset is "+str(test_data.shape))

#ratio of train to test data 
num = (train_data.shape[0]/(train_data.shape[0]+test_data.shape[0]))*100
den= (test_data.shape[0]/(train_data.shape[0]+test_data.shape[0]))*100
print("The ratio of test to train is " + str(round(num))+":"+str(round(den)))

#check for mean, median, max and min length of test and train dataset
#train data 

print("The mean length of tweets for train data "+ str(round(train_data['text'].str.len().mean())))
print("The median length of tweets for train data "+ str(round(train_data['text'].str.len().median())))
print("The min length of tweets for train data "+ str(train_data['text'].str.len().min()))
print("The max length of tweets for train data "+ str(train_data['text'].str.len().max()))

#test data

print("The mean length of tweets for test data "+ str(round(test_data['text'].str.len().mean())))
print("The median length of tweets for testdata "+ str(round(test_data['text'].str.len().median())))
print("The min length of tweets for test data "+ str(test_data['text'].str.len().min()))
print("The max length of tweets for test data "+ str(test_data['text'].str.len().max()))

# We can observe there is a little skewness of data towards the left. 
# Also the mean and median of both test and train data are equal 

# see length of train and test data by this histogram

length_train_dataset = train_data['text'].str.len()
length_test_dataset = test_data['text'].str.len()
plt.hist(length_train_dataset, bins=20,label="Train text")
plt.hist(length_test_dataset, bins=20,label="Test text")
plt.legend() 
plt.show()

clean_test =pd.read_csv('./cleaned_test.csv')
clean_train =pd.read_csv('./cleaned_train.csv')
combine = pd.concat([clean_train,clean_test])
#plotting all  words in wordcloud
all_words = ' '.join([text for text in combine['text_new']]) 
from wordcloud import WordCloud
wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#plotting for republican
# 0 is Republican
normal_words= ' '.join([text for text in combine['text_new'][combine['labels']==0]])
wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

# 1 is Democrat
negative_words= ' '.join([text for text in combine['text_new'][combine['labels']==1]])
wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

#checking for balance in the training dataset 
plt.hist(train_data["labels"], bins=10, color='blue', edgecolor='black', alpha=0.5)
plt.title('Histogram of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
#This shows there are almost equal number of both labels in teh traing dataset


