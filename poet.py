#!/usr/bin/env python
"""
@author: phdenzel

A machine-learning text-generation poet
"""
import sys
import numpy as np
# import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from makedir import mkdir_p

# Read the text
textfile = "sonnets.txt"
print("Gathering data/reading text: '{}'".format(textfile))
filepath = "/".join([sys.path[0], textfile])
sonnets = open(filepath, mode='r', encoding='utf-8-sig').read()
text = sonnets.lower()

characters = sorted(list(set(text)))
numb_map = {n: char for n, char in enumerate(characters)}
char_map = {char: n for n, char in enumerate(characters)}

# Turn text into pre-processed data
data = []   # training data
truth = []  # "truth" data
len_seq = 100
print("Pre-processing data...")
for i in range(0, len(text)-len_seq, 1):
    sequence = text[i:i + len_seq]
    label = text[i+len_seq]
    data.append([char_map[char] for char in sequence])
    truth.append(char_map[label])
# reshape to (number of sequences, length of sequence, number of features) & normalize
data = np.reshape(data, (len(data), len_seq, 1)) / float(len(characters))
# one-hot encode
onehot = np_utils.to_categorical(truth)

# Build model
#     - sequential
#     - two long short-term memory layers <lstm1, lstm2> w/ <N> units
#     - dropout rate <do_rate> 20%
#     - output layer: one-hot encoded character vector
N = 400
do_rate = 0.2
model = Sequential()
lstm1 = LSTM(N, input_shape=(data.shape[1], data.shape[2]), return_sequences=True)
lstm2 = LSTM(N)
dropout1 = Dropout(0.2)
dropout2 = Dropout(0.2)
output = Dense(onehot.shape[1], activation='softmax')
print("Building model...")
model.add(lstm1)
model.add(dropout1)
model.add(lstm2)
model.add(dropout2)
model.add(output)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Fit model
nepochs = 1
bsize = 100
savedir = "/".join([sys.path[0], "models/"])
savef = "ollam_{N:}_{do:}_{N:}_{do:}_{bs:}.h5".format(
    N=N, do=do_rate, bs=bsize)
filename = savedir+savef
print("Fitting models...")
model.fit(data, onehot, epochs=nepochs, batch_size=bsize)
print("Saving model fits: '{}'".format(filename))
mkdir_p(savedir)
model.save_weights(filename)
model.load_weights(filename)
