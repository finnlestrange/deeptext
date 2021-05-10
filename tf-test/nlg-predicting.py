# BIG WIP - does not currently function

import tensorflow as tf
import os
import sys
import time
import numpy

# Saves the location of the saved model to a variable
model_location = str(sys.argv[1])


# Necesary functions for converting to and from ids and chars



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

