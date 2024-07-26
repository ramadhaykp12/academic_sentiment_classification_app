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
import pickle
import json
import re

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

@st.cache(allow_output_mutation=True)
def Load_model():
    # Load model
    model = load_model('cnn_model.h5')
    session = K.get_session()
    return model, session
    
# Streamlit app
st.title('Aplikasi Klasifikasi Teks')

input_text = st.text_area('Masukkan teks untuk klasifikasi:')
model, session = Load_model()
if st.button('Klasifikasi'):
    if input_text:
        K.set_session(session)
        # Preprocess text
        preprocessed_text = preprocess_text(input_text)
        # Prepare input for model
        input_data = prepare_input(preprocessed_text, tokenizer)
        # Predict
        prediction = model.predict(input_data)
        prediction_prob_negative = prediction[0][0]
        prediction_prob_neutral = prediction[0][1]
        prediction_prob_positive= prediction[0][2]
        prediction_class = prediction.argmax(axis=-1)[0]
        print(prediction.argmax())
        # Display result
        st.header('Prediction using LSTM model')
        if prediction_class == 0:
          st.warning('Thread has negative sentiment')
        if prediction_class == 1:
          st.success('Thread has neutral sentiment')
        if prediction_class==2:
          st.success('Thread has positive sentiment')
