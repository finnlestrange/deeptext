# Keras Text PreProcessing Test

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000
embedding_dim = 32
max_length = 16
trunc_type = 'post'
padding_type = 'post'
# https://www.kaggle.com/hamishdickson/using-keras-oov-tokens - OOV token necessary for filtering out rare words and
# misspellings and variations of other words
#oov_tok = "<OOV>"
training_size = 20000

sentences = [
    "Test sentence 1",
    "Test sentence 2",
    "This is another test sentence"
]

tokenizer = Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding=padding_type, truncating=trunc_type, maxlen=5)

print(word_index)
print(sequences)
print(padded)
