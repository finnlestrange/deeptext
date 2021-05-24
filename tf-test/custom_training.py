import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
import sys
import string
from copy import deepcopy

# Loading variables from script arguments - error if not enough args
"""
data_dir = directory where dataset is kept
model_name = name of the model/directory of the model
dataset_size = number of files to be imported from dataset
epochs = number of epochs model will train for
"""
if len(sys.argv) not in [2, 3, 4]:
    sys.exit("Usage: python custom_training [data_dir] [model_name] [epochs]")

data_dir = str(sys.argv[1])
model_name = str(sys.argv[2])
epochs = int(sys.argv[3])

# Majority of classes and functions defined
"""
load_data --> Loads data from dataset
split_input_target --> Splits dataset into input and target texts for training
trainingModel class --> Model class for training the AI
text_from_ids --> Converts char ids into text for readability
WIP - Perhaps make dataset creation a function in the future
"""


def load_data(location):
    # List for holding the sentences from the entirety of the dataset
    reviews = []

    # Loading all the sentences from the dataset
    for file in os.listdir(location):
        with open(os.path.join(location, file), "r+", encoding="UTF-8") as f:
            reviews.append(f.read())

    return reviews


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


class trainingModel(tf.keras.Model):
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


class customTraining(trainingModel):
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {'loss': loss}


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# Making a model directory and implementing checkpoints for the model
os.mkdir(model_name)

checkpoint_dir = os.path.join(model_name, "Checkpoints")
os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Dataset formatting
"""
Loads data from the dataset files and converts into a single string for testing.
Each new line of the string is a different review from the dataset.
Data is then split into testing and training groups, and then is shuffled and
prefetched.
"""
# Data loading
reviews = load_data(data_dir)
reviews_string = ""
for a in reviews:
    reviews_string += str(a + "\n")
vocab = sorted(set(reviews_string))

# Data formatting - ensuring no invalid (AKA endian) chars are found in dataset
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

# Creating methods and functions to convert to and from char representations and text
chars = tf.strings.unicode_split(reviews_string, input_encoding="UTF-8")
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)
chars_from_ids = preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary()
)

# Turning word ids into datasets
ids_dataset = tf.data.Dataset.from_tensor_slices(ids_from_chars(chars))

# Defining the sequences length for each training/testing sequence and creating said
# sequences
seq_length = 100
examples_per_epoch = len(vocab)//(seq_length+1)
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# Finally creating the actual dataset and batching it
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(10000)  # Stops mem leak
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Creating the training model
"""
Creating the training model itself and training it for [epochs] epochs
embedding_dim --> Size of the input layer - maps char ids for model use
rnn_units --> Size of the RNN layer
vocab_size --> Size of the output layer - each logit represents a char
"""
embedding_dim = 256
rnn_units = 1024
vocab_size = len(vocab)

# Object creation
model = customTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# Loss and optimizer functions for compilation - adam is nice for efficiency
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss)

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

# Saving done here to use in prediction algo
model.save_weights(os.path.join(model_name, f"{model_name}_model"))

# WIP - Work on a method to save crucial dataset info so that the prediction algo
# doesn't have to
with open(os.path.join(model_name, "model_details.txt"), "w", encoding="UTF-8") as file:
    file.write("Model details-\n")
    file.write(f"Model name: {model_name}\n")
    file.write(f"Dataset size: Full\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Training type: One step\n")
