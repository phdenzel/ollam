import os
import tensorflow as tf

import ollam.utils
import ollam.models

encoding = 'UTF-8'
word_chars = 'abcdefghijklmnopqrstuvwxyz'
punctuation_chars = ' .,:;?!'
special_chars = '()[]&$%/'
default_encoder = tf.keras.layers.StringLookup(vocabulary=sorted(set(word_chars)))
default_decoder = tf.keras.layers.StringLookup(vocabulary=default_encoder.get_vocabulary(),
                                               invert=True, mask_token=None)
one_letter_words = ['a', 'i']
two_letter_words = ['to', 'in', 'it', 'is', 'be', 'as', 'at', 'so', 'we', 'he',
                    'by', 'or', 'on', 'do', 'if', 'me', 'my', 'up', 'an', 'go',
                    'no', 'us', 'am']


HOME_DIR = os.path.expanduser("~")
DOT_DIR = os.path.join(HOME_DIR, ".ollam")
DATA_DIR = os.path.join(DOT_DIR, "data")
MDL_DIR = os.path.join(DOT_DIR, "models")
LOG_DIR = os.path.join(DOT_DIR, "log")
TMP_DIR = os.path.join(DOT_DIR, "tmp")


sequence_length = 100
optimizer = 'Adam'
learning_rate = 0.01
dropout_rate = 0.2
epochs = 50
batch_size = 64
buffer_size = 10000
validation_split = 0.1
