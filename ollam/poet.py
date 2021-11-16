#!/usr/bin/env python
"""
@author: phdenzel

A machine-learning poet
"""

import sys
import os
import random
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from makedir import mkdir_p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Ollam(object):
    """
    Ollam:
        'An ollam, or ollamh (anglicised as ollave or ollav),
        in early Irish Literature, is a member of the highest rank of fili,
        an elite order of poets in Ireland.'
    """
    def __init__(self, filepath, sequence_length=100, lstm_layers=2, units=400, dropout_rate=0.2,
                 learning_cycles=1, batch_size=100, verbose=False):
        """
        Initialize a poet

        Args:
            filepath <str> - the file path to the poet's learning material

        Kwargs:
            sequence_length <int> - length of character sequence used for training
            lstm_layers <int> - number of layers in the network
            units <int> - number of neurons per layer in the network
            dropout_rate <float> - a dropout rate with which neurons are destroyed
            learning_cycles <int> - number of epochs in the learning process
            batch_size <int> - number of steps with which the learning proceeds
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Ollam object> - standard initializer
        """
        # validate input
        if not isinstance(filepath, str):
            raise TypeError("Input path needs to be string")
        if '~' in filepath:
            filepath = os.path.expanduser(filepath)
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            try:               # python 3
                FileNotFoundError
            except NameError:  # python 2
                FileNotFoundError = IOError
            raise FileNotFoundError("'{}' does not exist".format(filepath))
        if True not in [filepath.endswith(ext) for ext in ('.txt', '.text')]:
            raise ValueError('Input file needs to be a .txt file')
        self.book = os.path.basename(filepath)
        self.library = os.path.dirname(filepath)

        # read the text
        self.text = open(filepath, mode='r', encoding='utf-8-sig').read()
        self.text = self.text.lower()
        self.alphabet = sorted(list(set(self.text)))
        self.state = {}
        self.state['char_map'] = {char: n for n, char in enumerate(self.alphabet)}
        self.state['num_map'] = {n: char for n, char in enumerate(self.alphabet)}

        # turn text into learnable data
        self.state['data'] = []        # training data
        self.state['truth'] = []  # "truth" data
        self.state['sequence_length'] = sequence_length
        for i in range(len(self.text)-self.state['sequence_length']):
            seq = self.text[i:i + self.state['sequence_length']]
            cat = self.text[i + self.state['sequence_length']]
            self.state['data'].append([self.state['char_map'][char] for char in seq])
            self.state['truth'].append(self.state['char_map'][cat])

        # reshape to (number of sequences, length of sequence, number of features)
        shape = (len(self.state['data']), self.state['sequence_length'], 1)
        self.state['data_norm'] = np.reshape(self.state['data'], shape)
        self.state['data_norm'] = self.state['data_norm'] / float(len(self.alphabet))
        self.state['one_hot'] = np_utils.to_categorical(self.state['truth'])

        # build default network
        self.brain(lstm_layers=lstm_layers, units=units, dropout_rate=dropout_rate)
        # learn from the text
        self.learn(learning_cycles=learning_cycles, batch_size=batch_size, read_only=True)

        # some verbosity
        if verbose:
            print(self.__v__)

    @property
    def _tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['mind', 'book', 'library', 'alphabet']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of Ollam attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                          for t in self._tests])

    def brain(self, lstm_layers=2, units=400, dropout_rate=0.2, verbose=False):
        """
        Construct the poets mind, i.e. build a
        network model:
            - sequential
            - [2] long short-term memory layers w/ [400] units
            - dropout rate of [20]%
            - output layer: one-hot encoded character vector

        Args:
            None

        Kwargs:
            lstm_layers <int> - number of layers in the network
            units <int> - number of neurons per layer in the network
            dropout_rate <float> - a dropout rate with which neurons are destroyed
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        self.state['lstm_layers'] = lstm_layers
        self.state['units'] = units
        self.state['dropout_rate'] = dropout_rate
        self.mind = Sequential()
        inp_shape = (self.state['data_norm'].shape[1], self.state['data_norm'].shape[2])
        self.mind.add(LSTM(units, input_shape=inp_shape, return_sequences=True))
        self.mind.add(Dropout(dropout_rate))
        for i in range(1, lstm_layers-1):
            self.mind.add(LSTM(units, return_sequences=True))
            self.mind.add(Dropout(dropout_rate))
        self.mind.add(LSTM(units))
        self.mind.add(Dropout(dropout_rate))
        self.mind.add(Dense(self.state['one_hot'].shape[1], activation='softmax'))
        self.mind.compile(loss='categorical_crossentropy', optimizer='adam')
        if verbose:
            print("brain".ljust(20)
                  + "\t{} LSTM layers of {} units and dropout rate of {}".format(
                lstm_layers, units, dropout_rate))

    def learn(self, learning_cycles=None, batch_size=None, read_only=False, write=True,
              verbose=False):
        """
        Make the poet learn from the data he collected

        Args:
            None

        Kwargs:
            learning_cycles <int> - number of epochs in the learning process
            batch_size <int> - number of steps with which the learning proceeds
            write <bool> - save the learning weights in the models/ directory
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        if learning_cycles is not None and isinstance(learning_cycles, int):
            self.state['learning_cycles'] = learning_cycles
        if batch_size is not None and isinstance(batch_size, int):
            self.state['batch_size'] = batch_size
        self.state['models_directory'] = "/".join([sys.path[0], "models/"])
        self.state['ollam_h5'] = "ollam_"
        self.state['ollam_h5'] += "_".join(["{N:}_{do:}".format(
            N=self.state['units'],
            do=self.state['dropout_rate']) for i in range(self.state['lstm_layers'])])
        self.state['ollam_h5'] += "_{bs:}".format(bs=self.state['batch_size'])
        self.state['ollam_h5'] += "x{lc:}.h5".format(lc=self.state['learning_cycles'])
        filename = self.state['models_directory'] + self.state['ollam_h5']
        if os.path.exists(filename):
            self.mind.load_weights(filename)
        elif not read_only:
            self.mind.fit(self.state['data_norm'], self.state['one_hot'],
                          epochs=self.state['learning_cycles'],
                          batch_size=self.state['batch_size'])
            if write:
                mkdir_p(self.state['models_directory'])
                self.mind.save_weights(filename)
        if verbose:
            print("Training of {} cycle(s) completed -> models/{}".format(
                self.state['learning_cycles'], self.state['ollam_h5']))

    def speak(self, length=400, random_seed=1, verbose=True):
        """
        Let the poet speak
        (i.e. start at a random point in the text and predict what characters should come next)

        Args:
            None

        Kwargs:
            length <int> - number of characters the speech has
            random_seed <int> - seed for the random generator which determines where to start
            verbose <bool> - verbose mode; print command line statements

        Return:
            speech <str> - the ramblings of the poet
        """
        random.seed(random_seed)
        random_start = random.randint(0, len(self.state['data']))
        char_mapped = self.state['data'][random_start]
        chars = [self.state['num_map'][value] for value in char_mapped]
        for i in range(length):
            mapped = np.reshape(char_mapped, (1, len(char_mapped), 1))
            mapped = mapped / float(len(self.alphabet))
            pred_key = np.argmax(self.mind.predict(mapped, verbose=0))
            # seq = [self.state['num_mapping'][value] for value in char_mapped]
            chars.append(self.state['num_map'][pred_key])
            char_mapped.append(pred_key)
            char_mapped = char_mapped[1:]
        speech = "".join(chars)
        if verbose:
            print("Speech:\n"+speech)
        return speech


def parse_args():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("textfile", nargs='?',
                        help="Input path to learning material text file",
                        default=sys.path[0]+'/sonnets.txt')

    parser.add_argument("-s", "--sequence_length", metavar="<length>", type=int,
                        help="length of character sequence used for training")
    parser.add_argument("-l", "--lstm-layers", metavar="<n_layers>", type=int,
                        help="number of layers in the network")
    parser.add_argument("-n", "--units", metavar="<n_units>", type=int,
                        help="number of neurons per layer in the network")
    parser.add_argument("-d", "--dropout-rate", metavar="<rate>", type=int,
                        help="a dropout rate with which neurons are destroyed")
    parser.add_argument("-e", "--learning-cycles", metavar="<n_cycles>", type=int,
                        help="number of epochs in the learning process")
    parser.add_argument("-b", "--batch-size", metavar="<n_steps>", type=int,
                        help="number of steps with which the learning proceeds")

    parser.add_argument("--verbose-off", dest="verbose", action="store_false",
                        help="run program in verbose mode", default=True)

    args = parser.parse_args()
    textfile = args.textfile
    delattr(args, 'textfile')
    return textfile, {k: args.__dict__[k] for k in args.__dict__.keys()
                      if args.__dict__[k] is not None}


if __name__ == "__main__":
    textfile, kwargs = parse_args()
    poet = Ollam(textfile, **kwargs)
    poet.learn(verbose=True)
    poet.speak()
