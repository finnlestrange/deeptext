import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
import sys
import time
import string
from copy import deepcopy

# Loading variables from script arguments
"""
data_dir = directory where the model is being kept
model_name = name of the model directory to access
"""
if len(sys.argv) not in [2, 3, 4]:
    sys.exit("Usage: python predicting.py [data_dir] [model_name] [text_size]")

data_dir = str(sys.argv[2])
model_name = str(sys.argv[3])
text_size = int(sys.argv[4])

# Prompts user for a starting string to feed to the model
starting_string = str(input("Please input a starting string for the model: "))

# Also finds the details of the specified model and finds dataset size
with open(os.path.join(model_name, "model_details.txt"), "r+") as file:
    lines = file.readlines()
    dataset_size = int(lines[2].replace("Dataset size: ", ""))

print(dataset_size)
# raise NotImplementedError
# Majority of classes and functions defined
"""
load_data --> Loads dataset data from specified directory
trainingModel --> Class for the training model, needed for loading
predictingModel --> Class for the predicting model, predicts next char
"""


def load_data(location):
    # List for holding the sentences from the entirety of the dataset
    reviews = []

    # Loading all the sentences from the dataset
    for file in os.listdir(location):
        with open(os.path.join(location, file), "r+", encoding="UTF-8") as f:
            reviews.append(f.read())

    return reviews


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


class predictingModel(tf.keras.Model):
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
        # Apply the prediction mask: prevents "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


# Dataset importing
"""
Dataset needs to be imported due to the method in which model saving is done,
actual tf model saving does not work so weights are loaded into a training
model class instead. Then the prediction model is created and run using next
chars as input. Dataset is imported to provide chars_from_ids and ids_from_chars.
"""
# Data loading
reviews = load_data(data_dir)
reviews_string = ""
for n in range(dataset_size):
    reviews_string += (str(reviews[n]) + "\n")
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

# Creating the necessary methods for the prediction model
chars = tf.strings.unicode_split(reviews_string, input_encoding="UTF-8")
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)
chars_from_ids = preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary()
)

# Model creation
"""
Both training and prediction models are created. Weights are loaded from saved weights.
Model layer guide-
embedding_dim --> Size of the input layer - maps char ids for model use
rnn_units --> Size of the RNN layer
vocab_size --> Size of the output layer - each logit represents a char
---
For more info see the training script
"""
# Model layers vars
embedding_dim = 256
rnn_units = 1024
vocab_size = len(vocab)

# Building training model
model = trainingModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# Importing weights from prior training
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimzer="adam", loss=loss)
model.load_weights(os.path.join(model_name, f"{model_name}_model"))

# Building predicting model
predicting_model = predictingModel(model, chars_from_ids, ids_from_chars)

# Actual prediction
"""
Model takes an input string and continually feeds itself characters as inputs. Each
produced character is re-used as the input for the next predicted character. Creating
text.
"""
start = time.time()
states = None
next_char = tf.constant([starting_string])

result = [next_char]

for n in range(text_size):
    next_char, states = predicting_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode("utf-8"))
with open(os.path.join(model_name, f"{model_name}_output.txt"), "w", encoding="UTF-8") as file:
    for a in range(len(result)):
        file.write(f"Sentence {a+1}:\n")
        file.write(str(result[a].numpy().decode("utf-8")))
        file.write("\n")
        file.write("\n")
