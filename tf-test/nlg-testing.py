# Importing tf and required libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import time

# Get a dataset later - and import vectorized representations of tokens
"""
Data setup here
all_word_ids = list of all vectorized words
words_in_text = Number of words in text
"""
all_word_ids = None
words_in_text = 0

# Slicing up the word ids into datasets
ids_dataset = tf.data.Dataset.from_tensor_slices(None)

# Prediction
"""
Write blurb about prediction
"""

# Setting sequence length to 15 - average sentence length
seq_length = 15
examples_per_epoch = words_in_text//(seq_length+1)

# Converting list of words into desired sequence lengths
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


# Splits dataset into input and target texts for training
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