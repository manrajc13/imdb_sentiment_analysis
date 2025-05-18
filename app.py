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
  sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"

  return sentiment, prediction[0][0]


### Streamlit app

import streamlit as st

# Set page config
st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", layout="centered", page_icon="🎥")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("## 🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Use this app to find out if a movie review expresses a **positive** or **negative** sentiment using a pretrained RNN model.")

# Input box
st.markdown("### 📝 Enter a movie review below:")
user_input = st.text_area(" ", "The movie was fantastic! I loved it. 👌")

# Predict button
if st.button("🔍 Predict Sentiment"):

    sentiment, confidence = predict_sentiment(user_input)
    
    st.success(f"**Sentiment:** {sentiment}")
    st.info(f"**Confidence Score:** {confidence:.2f}")

# Add explanation with an expander
with st.expander("ℹ️ How to interpret results"):
    st.write("""
    - A **confidence score > 0.5** suggests **positive** sentiment.
    - A **score < 0.5** indicates **negative** sentiment.
    - This app uses a Recurrent Neural Network (RNN) trained on the IMDB dataset.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")


