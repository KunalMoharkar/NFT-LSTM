from django.shortcuts import render
from django.http import HttpResponse
from numpy.core.arrayprint import ComplexFloatingFormat
# Create your views here.
import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from nltk.corpus import stopwords   # to get collection of stopwords
from sklearn.model_selection import train_test_split       # for splitting dataset
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.layers import Embedding, LSTM, Dense # layers of the architecture
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.models import load_model   # load saved model
import re


def index(request):

    loaded_model = load_model('/home/kdm1700/Desktop/gitwala/sentiment-analysis-IMDB-Review-using-LSTM/Textpad/models/LSTM.h5')
    english_stops = set(stopwords.words('english'))
    token = Tokenizer(lower=False)
    max_length = 130
   

    if request.method == 'POST':

        review = request.POST['review']

    

        regex = re.compile(r'[^a-zA-Z\s]')
        review = regex.sub('', review)
        #print('Cleaned: ', review)

        words = review.split(' ')
        filtered = [w for w in words if w not in english_stops]
        filtered = ' '.join(filtered)
        filtered = [filtered.lower()]

        #print('Filtered: ', filtered)

        tokenize_words = token.texts_to_sequences(filtered)
        tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
        #print(tokenize_words)

        result = loaded_model.predict(tokenize_words)

        if result >= 0.7:
            res = 'positive'
        else:
            res = 'negative'

        context = {'res':result, 'review':review}
    
        return render(request, 'home/index.html', context)

    return render(request, 'home/index.html')

