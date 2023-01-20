#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 08:48:45 2023

@author: kaoutherboulaouinat"""

import streamlit as st
import pickle 
import sys
import time 
from gensim.models import Word2Vec 
from keras.models import load_model 

from keras_preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences 



from PIL import Image

st.set_page_config(page_title="Sentiment Analysis App", page_icon="🔗",
                   layout="wide")  # needs to be the first thing after the streamlit import
# load lstm model

#model = load_model('../model.h5') 
model = load_model('/app/app/model.h5')
w2v_model = Word2Vec.load('/app/app/model.w2v')

with open('/app/app/tokenizer.pkl', 'rb') as handle: tokenizer = pickle.load(handle) 
with open('/app/app/encoder.pkl', 'rb') as handle: encoder = pickle.load(handle)

SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024


POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]: label = NEGATIVE 
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
    
def predict(text, include_neutral=True): 
    start_at = time.time() 
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH) 
    # Predict 
    score = model.predict([x_test])[0] 
    # Decode  sentiment 
    label = decode_sentiment(score, include_neutral) 
    return {"label": label, "score": float(score), "elapsed_time": time.time()-start_at}









st.title("Sentiment Analysis App with LSTM")
st.subheader("Predict a sentiment of a sentence using Predict, or click on graphs to display plots")
st.sidebar.title(":chart_with_upwards_trend: Prediction")
text_input = st.sidebar.text_input(
        "Enter some text 👇",
        placeholder="Sentence to predict",
    )
col1, col2, col3 = st.sidebar.columns(3)
if col2.button('Predict'):
    print(type(text_input))
    result=predict(text_input,False)
    st.subheader(":red[**LSTM Prediction**] :sunglasses:")
    st.markdown(f'  - **Label**: :green[{result["label"]}] \n  - **Score**: :green[{result["score"]}] \n  - **Elapsed time**: :green[{result["elapsed_time"]}]',unsafe_allow_html=True)
             



            

st.sidebar.title(":chart_with_downwards_trend: Display plots")   
col11, col12, col13 = st.sidebar.columns(3)
if col12.button('Graphs'):
     st.subheader(":red[LSTM]")
     col21, col22, col23 = st.columns(3)  
     lstm1 = Image.open('/app/app/plot1.png')
     lstm2 = Image.open('/app/app/plot.png')
     lstm3 = Image.open('/app/app/matrix.png')
     col21.image(lstm1, caption='accuracy')
     col22.image(lstm2, caption='loss')
     col23.image(lstm3, caption='confusion matrix')



