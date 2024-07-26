import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras import backend as K
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
import json
import re

model = load_model('cnn_model.h5')
# Inisialisasi tokenizer yang sama digunakan saat melatih model
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Streamlit app
st.title('Aplikasi Klasifikasi Teks')

input_text = st.text_input('Masukkan teks untuk klasifikasi:')
if st.button('Klasifikasi'):
    if input_text:
    sequences = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded)

    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_label = sentiment_labels[np.argmax(prediction)]

    st.write(f"Predicted Sentiment: **{predicted_label}**")
