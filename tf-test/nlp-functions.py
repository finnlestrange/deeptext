import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


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
    tokenizer = Tokenizer(num_words = 1000, oov_token="<OOV>")

    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)

    padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

    return padded, word_index


def average_sentence_len(sentences):
    count = 0
    for sentence in sentences:
        list = sentence.split()
        count += len(list)
    return count//len(sentences)

# sentences = load_data("C:\\Users\\nicol\\PycharmProjects\\deep-text\\tf-test\\data\\pos")
# print(average_sentence_len(sentences))
