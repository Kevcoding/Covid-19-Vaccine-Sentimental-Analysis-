import streamlit as st

#Data Manipulation
import pandas as pd
import numpy as np

#Data Preprocessing
import random
import nltk
import string
import re
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


#Sentimental Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import neattext.functions as nfx


#Machine Learning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Data visualization
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns

header = st.container()
dataset = st.container()
features= st.container()
model_training = st.container()

with header:
	st.title('Sentimental Analysis of Covid-19 Tweets')


with dataset:
	st.header('Covid-19 Vaccine Tweets Dataset')
	st.text('This project looks into the Sentimental Analysis of Covid-19 Twitter tweets')
	st.text('The first few entries of our dataset are')
	
	df = pd.read_csv("vaccination_tweets.csv")
	st.write(df.head(10))
	
	st.text('Duplicates Check')
	st.write(df.duplicated(keep=False).sum())
	
	st.text('Dropping Duplicates')
	st.write(df.drop_duplicates())
	st.text('Missing values check')
	st.write(df.isnull().sum())
	
	st.text('Source of dataset')
	st.write(df["source"].value_counts().nlargest(10))
	st.subheader('Bar plot of sources')
	st.bar_chart(df['source'].value_counts().nlargest(20))

	st.text('Data Encoding')
	df["user_verified"] = df["user_verified"].astype(int)




with features:
	st.header('Our features')
	st.markdown('* **first feature:** id')
	st.markdown('* **second feature:** user_name')
	st.markdown('* **third feature:** user_location')
	st.markdown('* **fourth feature:** user_description')
	st.markdown('* **fifth feature:** user_created')
	st.markdown('* **sixth feature:** user_followers')
	st.markdown('* **seventh feature:** user_friends')
	st.markdown('* **eighth feature:** user_favourites')
	st.markdown('* **nine feature:** user_verified')
	st.markdown('* **ten feature:** text')
	st.markdown('* **eleveth feature:** hashtags')
	st.markdown('* **twelveth feature:** source')
	st.markdown('* **thirteen feature:** retweets')
	st.markdown('* **fourteen feature:** favorites')
	st.markdown('* **fifteen feature:** is_retweet')


with model_training:
	st.header('Training our model')
	st.text('Choice of Model Parameters and variation in performance')
	
	sel_col, disp_col = st.columns(2)
	
	max_depth = sel_col.slider('What is your max_depth of model?', min_value=10, max_value=100, value=20, step=10)
	
	n_estimators = sel_col.selectbox('How many estimators are required?', options=[100,200,300,'No limit'],index=0)
	
	input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'text')


	




