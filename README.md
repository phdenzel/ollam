
# Table of Contents

1.  [Prerequisites](#org8416431)
2.  [Usage](#orgf293916)

An attempt to use Natural Language Processing to create an 'artificial poet'.
Just for fun, I used William Shakespeare's material to train the machine from [http://www.gutenberg.org/ebooks/1041?msg=welcome<sub>stranger</sub>](http://www.gutenberg.org/ebooks/1041?msg=welcome_stranger).


<a id="org8416431"></a>

# Prerequisites

This program uses following python modules

-   keras
-   h5py
-   tensorflow
-   numpy


<a id="orgf293916"></a>

# Usage

    usage: ollam [-h] [-t] [-l <length>] [--optimizer <optimizer>] [--lr <rate>] [-d <rate>] [--epochs <epochs>] [--bs <size>] [--vs <ratio>]
    	     [--test] [-v <level>]
    
    optional arguments:
      -h, --help            show this help message and exit
      -t, --train, --train-mode
    			Run ollam in train mode.
      -l <length>, --length <length>, --sequence-length <length>
    			Length of character sequence used for training.
      --optimizer <optimizer>
    			Optimizer function classname for Tensorflow model.
      --lr <rate>, --learning-rate <rate>
    			Learning rate for Tensorflow optimization.
      -d <rate>, --dropout-rate <rate>
    			Dropout rate for model training.
      --epochs <epochs>     Number of epochs for model fitting.
      --bs <size>, --batch-size <size>
    			Batch size for model fitting.
      --vs <ratio>, --val-split <ratio>, --validation-split <ratio>
    			Validation split ratio for monitoring training session.
      --test, --test-mode   Run program in testing mode.
      -v <level>, --verbose <level>
    			Define level of verbosity

