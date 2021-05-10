# This file combines training and predicting into one broad file - trains and then outputs data - inefficient
# Fix imports at some point
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import os
import time
import sys
import string
import numpy
from copy import deepcopy

# oh well no imports, pasting functions in here instead
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
"""
def vectorize(sentences):
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")

    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)

    padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

    return padded, word_index
"""

# Directory where the dataset can be found
locations = str(sys.argv[1])

# Directory where training checkpoints can be saved
checkpoint_dir = str(sys.argv[2])
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# Dataset imported - IMDB reviews data
"""
Data setup here
all_word_ids = list of all vectorized words
words_in_text = Number of words in text
"""
sentences = load_data(locations)
# Convert sentences into a single string but with each line being a different review
sentence = ""
# Full dataset is 2519*5 sentences, original training will be on only 50 cuz all of them is way too much
for a in range(50):
    sentence += str(sentences[a])
vocab = sorted(set(sentence))
accepted_chars = [" ", "\n"]
for letter in string.ascii_letters:
    accepted_chars.append(letter)
for digit in string.digits:
    accepted_chars.append(digit)
for punctuation in string.punctuation:
    accepted_chars.append(punctuation)
for char in deepcopy(vocab):
    if char not in accepted_chars:
        vocab.remove(char)
# all_word_ids, word_index = vectorize(sentences)
vocab = sorted(set(sentence))
chars = tf.strings.unicode_split(sentence, input_encoding="UTF-8")
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
words_in_text = 1000

# Slicing up the word ids into datasets
all_ids = ids_from_chars(chars)
# print(all_ids)
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# Prediction
"""
Write blurb about prediction
"""

# Setting sequence length to 15 - average sentence length - 228
seq_length = 100
examples_per_epoch = words_in_text//(seq_length+1)

# Converting list of words into desired sequence lengths
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
# for seq in sequences.take(1):
#     print(chars_from_ids(seq))


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

# DO NOT USE VARIABLES FOR RESPRESNTING SHUFFLE BUFFER OR BATCH SIZE - BREAKS EVERYTHING
# Dataset batching and shuffling and prefetching
dataset = dataset.shuffle(10000)
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Building a training model
"""
Write blurb about training models
"""

# Initialize variables for model
# Output layer size - equivalent to vocab size
vocab_size = len(vocab)

# Embedding dimension - Input layer - maps vectors
embedding_dim = 256

# Number of RNN units - middle layer
rnn_units = 1024


# This is the model object - taken from tf web
class MyModel(tf.keras.Model):
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
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


# Model object created - saved to var model
model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# Setting loss and optimizer functions - so far it seems that each step is 5 sentences - probably dataset stuff
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss)

history = model.fit(dataset, epochs=20, callbacks=[checkpoint_callback])


# Class for making a single-step prediction AKA predicting the next character - post-training
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


# Building the onestep model
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# Actually running the model
start = time.time()
states = None
next_char = tf.constant(["I saw"])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
with open("outputs/output.txt", "w", encoding="utf-8") as file:
    file.write(str(result[0].numpy().decode("utf-8")))

# Mandatory saving so I dont have to run this ever again
tf.saved_model.save(one_step_model, 'one_step')