# Political Orientation Classifier For Tweets

## Description & Motivation
According to Pew Research Center, roughly one-quarter of American adults use Twitter, and often it is used to share their views about politics and political issues. Twitter can have a huge impact on the political sphere in the US, and it can be helpful to determine the political orientation of certain tweets to identify the presence of Democratic vs. Republican voice on a social media platform and the issues each party cares the most about.

For this project, we are solving a text / document classification problem, and more specifically, we are creating a generative and a discriminative model to predict the political orientation (‘Democratic’ or ‘Republican’) of a tweet. 

## Requirements
To train and run the model, please include the following Python packages in your `requirements.txt` document:
```
pylint
black
numpy
pandas
matplotlib
transformers
datasets
nltk
torch
torchmetrics
sklearn
wordcloud
gensim
textblob
```

## Data
@Pragya to add more details about the dataset.


## Generative Model

# Description

We employed the naive bayes classifier as a generative model to classify tweets based on political affiliation. A naive bayes classifier is a machine learning model that uses model features to discriminate between different objects. This model is based on the bayes’ theorem, which assumes that all features are independent of each other, or in other words, the presence of a particular feature in a class is unrelated to the presence of any other feature.
