import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
import pickle
import json
import re

# Load model
model = load_model('cnn_model.h5')


# Inisialisasi tokenizer yang sama digunakan saat melatih model
tokenizer_file = 'tokenizer.json'
with open(tokenizer_file, 'rb') as handle:
    data = json.load(handle)
    tokenizer = tokenizer_from_json(data)

# Preprocessing functions
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove punctuation and stopwords
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def prepare_input(text, tokenizer, max_len=100):
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

# Streamlit app
st.title('Aplikasi Klasifikasi Teks')

input_text = st.text_area('Masukkan teks untuk klasifikasi:')
if st.button('Klasifikasi'):
    if input_text:
        # Preprocess text
        preprocessed_text = preprocess_text(input_text)
        # Prepare input for model
        input_data = prepare_input(preprocessed_text, tokenizer)
        # Predict
        prediction = model.predict(input_data)
        # Display result
        st.write('Prediksi:', np.argmax(prediction))
    else:
        st.write('Masukkan teks untuk klasifikasi.')
