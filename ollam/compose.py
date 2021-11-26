"""
ollam.compose

@author: phdenzel
"""
import sys
import numpy as np
import tensorflow as tf
import ollam
from ollam.models import ModelConfigurator


class ComposeStep(tf.keras.Model):
    def __init__(self, model, label_map, inference_map,
                 encoding=None):
        super().__init__()
        self.model = model
        self.label_map = label_map
        self.inference_map = inference_map
        self.encoding = ollam.encoding if encoding is None else encoding

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.label_map(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids), # -inf at each bad index
            indices=skip_ids,
            dense_shape=[len(self.label_map.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def step(self, inputs, states=None):
        input_chars = ollam.utils.split_string(inputs).to_tensor()
        input_ids = ollam.utils.encode_chars(
            input_chars,
            encoder=self.label_map,
            string_encoding=self.encoding)

        predicted_logits, states = self.model(inputs=input_ids,
                                              states=states,
                                              return_state=True)

        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.inference_map(predicted_ids)
        return predicted_chars, states

    def generate(self, num_steps, init_string, states=None):
        next_char = tf.constant([init_string])
        output = [next_char]
        for n in range(num_steps):
            next_char, states = self.step(next_char, states=states)
            output.append(next_char)
        output = ollam.utils.join_chars(output, encoding=self.encoding)
        return output


def compose(init_string=None, n_chars=None, verbose=True):
    if init_string is None:
        init_string = ollam.init_string
    texts = ollam.utils.load_texts(ollam.DATA_DIR, verbose=verbose)
    dataset, _ = ollam.models.ModelConfigurator.encode_text(
        texts[0], ollam.sequence_length)
    configurator = ollam.models.ModelConfigurator.from_archive(
        None, dataset=dataset, onehot_encode=False, verbose=verbose)
    configurator.load_model(from_checkpoints=True)
    configurator.validate(verbose=verbose)

    composer = ComposeStep(*configurator.compose_args)
    poem = composer.generate(ollam.n_chars, init_string)
    print(poem)


if __name__ == "__main__":

    configurator = ollam.models.load_from_archive('7A4M7OQBMJRNB')
    configurator.load_model(from_checkpoints=True)
    configurator.validate(verbose=True)

    composer = ComposeStep(*configurator.compose_args)
    poem = composer.generate(1000, 'THE LORD:')
    print(poem)
