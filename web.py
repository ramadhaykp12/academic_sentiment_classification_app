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
import json
import numpy as np

# Load model
model = load_model('cnn_model.h5')

# Inisialisasi tokenizer yang sama digunakan saat melatih model
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)


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
        # Prepare input for model
        input_data = prepare_input(input_text, tokenizer)
        # Predict
        prediction = model.predict(input_data)
        # Display result
        st.write('Prediksi:', prediction)
    else:
        st.write('Masukkan teks untuk klasifikasi.')
