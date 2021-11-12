
# Table of Contents

1.  [Prerequisites](#org1984af3)
2.  [Usage](#org632891c)

An attempt to use Natural Language Processing to create an 'artificial poet'.
Just for fun, I used William Shakespeare's material to train the machine from [http://www.gutenberg.org/ebooks/1041?msg=welcome<sub>stranger</sub>](http://www.gutenberg.org/ebooks/1041?msg=welcome_stranger).


<a id="org1984af3"></a>

# Prerequisites

This program uses following python modules

-   keras
-   h5py
-   tensorflow
-   numpy


<a id="org632891c"></a>

# Usage

    usage: poet.py [-h] [-s <length>] [-l <n_layers>] [-n <n_units>] [-d <rate>] [-e <n_cycles>] [-b <n_steps>] [--verbose-off] [textfile]
    
    @author: phdenzel
    
    A machine-learning poet
    
    positional arguments:
      textfile              Input path to learning material text file
    
    optional arguments:
      -h, --help            show this help message and exit
      -s <length>, --sequence_length <length>
    			length of character sequence used for training
      -l <n_layers>, --lstm-layers <n_layers>
    			number of layers in the network
      -n <n_units>, --units <n_units>
    			number of neurons per layer in the network
      -d <rate>, --dropout-rate <rate>
    			a dropout rate with which neurons are destroyed
      -e <n_cycles>, --learning-cycles <n_cycles>
    			number of epochs in the learning process
      -b <n_steps>, --batch-size <n_steps>
    			number of steps with which the learning proceeds
      --verbose-off         run program in verbose mode

