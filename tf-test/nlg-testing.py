# Importing tf and required libraries
# Literally the messiest and shittiest import ever but who the fuck cares
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time


# Fuck it no imports, pasting functions in here instead
# Function pastes:
# Conversion into functional form
def load_data(location):

    # List for holding the sentences from the entirety of the dataset
    sentences = []

    # Loading all the sentences from the dataset
    for file in os.listdir(location):
        with open(os.path.join(location, file), "r+", encoding="UTF-8") as f:
            sentences.append(f.read())

    return sentences


# Tokenization and conversion into vector matrices here
def vectorize(sentences):
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")

    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)

    padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

    return padded, word_index


locations = "C:\\Users\\nicol\\PycharmProjects\\deep-text\\tf-test\\data\\pos"

# Dataset imported - IMDB reviews data
"""
Data setup here
all_word_ids = list of all vectorized words
words_in_text = Number of words in text
"""
sentences = load_data(locations)
all_word_ids, word_index = vectorize(sentences)
words_in_text = 1000

# Slicing up the word ids into datasets
ids_dataset = tf.data.Dataset.from_tensor_slices(all_word_ids)

# Prediction
"""
Write blurb about prediction
"""

# Setting sequence length to 15 - average sentence length - 228
seq_length = 228
examples_per_epoch = words_in_text//(seq_length+1)

# Converting list of words into desired sequence lengths
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


# Splits dataset into input and target texts for training
@tf.autograph.experimental.do_not_convert
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# Finally creates the actual dataset
dataset = sequences.map(split_input_target)

# Training batches
"""
Write blurb about training batches
"""

# Batch size - explanation
BATCH_SIZE = 64

# Buffer size - stops memory leak
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Building a training model
"""
Write blurb about training models
"""

# Initialize variables for model
# Output layer size - equivalent to vocab size
vocab_size = len(all_word_ids)

# Embedding dimension - Input layer - maps vectors
embedding_dim = 256

# Number of RNN units - middle layer
rnn_units = 1024


# This is the model object - taken from tf web
class testModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_states=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# Model object created - saved to var model
model = testModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

model.summary()
# Model testing
# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions)