import streamlit as st

header = st.container()
dataset = st.container()
features= st.container()
model_training = st.container()

with header:
	st.title('Sentimental Analysis of Covid-19 Tweets')


with dataset:
	st.header('Covid-19 Vaccine Tweets Dataset')
	st.text('This project looks into the Sentimental Analysis of Covid-19 Twitter tweets')
	
	


with features:
	st.header('Our features')




with model_training:
	st.header('Training our VADER model')



