import os

import ollam.utils
import ollam.models


HOME_DIR = os.path.expanduser("~")
DOT_DIR = os.path.join(HOME_DIR, ".ollam")
DATA_DIR = os.path.join(DOT_DIR, "data")
MDL_DIR = os.path.join(DOT_DIR, "models")
LOG_DIR = os.path.join(DOT_DIR, "log")
TMP_DIR = os.path.join(DOT_DIR, "tmp")


sequence_length = 16
optimizer = 'Adam'
learning_rate = 0.001
epochs = 100
batch_size = 16
validation_split = 0.1
