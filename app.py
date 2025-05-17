import numpy as np 
import pandas as pd 
import streamlit as st
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# load the imdb dataset 
word_index = imdb.get_word_index()
reverse_word_index = {value : key for (key, value) in word_index.items()}


# load the pretrained model 

model = tf.keras.models.load_model('simple_rnn_imdb.h5')


# function to decode reviews 
def decoded_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to preprocess the input text
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# function to predict the sentiment of the input text


def predict_sentiment(review):
  preprocessed_input = preprocess_text(review) 

  prediction = model.predict(preprocessed_input)
  sentiment = "positive" if prediction > 0.5 else "negative"

  return sentiment, prediction[0][0]


## streamlit app
import streamlit as st 

st.title("IMDB Movie Review Sentiment Analysis App")
st.write("This app uses a pretrained RNN model to predict the sentiment of movie reviews from the IMDB dataset.")

# user_input 
user_input = st.text_area("Enter a movie review:", "The movie was fantastic! I loved it.")
if st.button("Predict Sentiment"):
  sentiment, confidence = predict_sentiment(user_input)
  st.write(f"Sentiment: {sentiment}")
  st.write(f"Confidence: {confidence:.2f}")

st.write("Note: A confidence score above 0.5 indicates a positive sentiment, while a score below 0.5 indicates a negative sentiment.")