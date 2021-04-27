import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "Test sentence 1",
    "Test sentence 2",
    "This is another test sentence"
]

tokenizer = Tokenizer(num_words = 100)

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

print(word_index)
print(sequences)
print(padded)