"""
ollam.__main__

@author: phdenzel
"""
import ollam


__all__ = ['main']


def arg_parse():
    from argparse import ArgumentParser, RawTextHelpFormatter
    p = ArgumentParser(prog='ollam', #description=__doc__,
                       formatter_class=RawTextHelpFormatter)

    p.add_argument("-c", "--compose",
                   dest="compose", metavar="<init_string>", type=str,
                   default='',
                   help="Run ollam in compose mode, based on initial input."
                   )
    p.add_argument("-n", "--n-chars", "--number-of-chars", "--number-of-characters",
                   dest="n_chars", metavar="<number>", type=int,
                   default=1000,
                   help="Number of characters for text generation."
                   )

    # Model training flags
    p.add_argument("-t", "--train", "--train-mode",
                   dest="train_mode", action="store_true", default=False,
                   help="Run ollam in train mode."
                   )
    p.add_argument("-l", "--length", "--sequence-length",
                   dest="sequence_length", metavar="<length>", type=int,
                   default=ollam.sequence_length,
                   help="Length of character sequence used for training."
                   )
    p.add_argument("--optimizer",
                   dest="optimizer", metavar="<optimizer>", type=str,
                   default=ollam.optimizer,
                   help="Optimizer function classname for Tensorflow model."
                   )
    p.add_argument("--lr", "--learning-rate",
                   dest="learning_rate", metavar="<rate>", type=float,
                   default=ollam.learning_rate,
                   help="Learning rate for Tensorflow optimization."
                   )
    p.add_argument("-d", "--dropout-rate",
                   dest="dropout_rate", metavar="<rate>", type=float,
                   default=ollam.dropout_rate,
                   help="Dropout rate for model training.")
    p.add_argument("--epochs",
                   dest="epochs", metavar="<epochs>", type=int,
                   default=ollam.epochs,
                   help="Number of epochs for model fitting."
                   )
    p.add_argument("--bs", "--batch-size",
                   dest="batch_size", metavar="<size>", type=int,
                   default=ollam.batch_size,
                   help="Batch size for model fitting."
                   )
    p.add_argument("--vs", "--val-split", "--validation-split",
                   dest="validation_split", metavar="<ratio>", type=float,
                   default=ollam.validation_split,
                   help="Validation split ratio for monitoring training session."
                   )
    
    p.add_argument("--test", "--test-mode", dest="test_mode", action="store_true",
                   help="Run program in testing mode.", default=False
                   )
    p.add_argument("-v", "--verbose", dest="verbose", metavar="<level>", type=int,
                   help="Define level of verbosity")

    args = p.parse_args()
    return p, args


def overwrite_dg_vars(args):
    """
    Args:
        args <Namespace>
    """
    ollam.init_string = args.compose
    ollam.n_chars = args.n_chars
    ollam.sequence_length = args.sequence_length

    ollam.optimizer = args.optimizer
    ollam.learning_rate = args.learning_rate
    ollam.epochs = args.epochs
    ollam.batch_size = args.batch_size
    ollam.validation_split = args.validation_split


def main():

    parser, args = arg_parse()
    overwrite_dg_vars(args)

    if args.train_mode:
        from ollam.models import train as main
    elif args.test_mode:
        from test import main
    elif args.compose:
        from ollam.compose import compose as main
    else:
        main = parser.print_help

    main()
    
