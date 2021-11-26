"""
ollam.utils

@author: phdenzel
"""
import os
import re
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

import ollam


def mkdir_p(pathname):
    """
    Create a directory as if using 'mkdir -p' on the command line

    Args:
        pathname <str> - create all directories in given path
    """
    from errno import EEXIST
    try:
        os.makedirs(pathname)
    except OSError as exc:  # Python > 2.5
        if exc.errno == EEXIST and os.path.isdir(pathname):
            pass
        else:
            raise


def camel_to_snake(name):
    """
    Convert from CamelCase to snake_case notation
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def generate_filename(prefix=None, name_id=None, part_no=None, extension='h5'):
    """
    Generate a unique filename for ollam model data

    Kwargs:
        prefix <str> - prefix of file name
        name_id <int> - ID to generate unique identifier
        part_no <int> - optional part number if landmarks part of a sequence
        extension <str> - file extension; default: npy (binary numpy extension)
    """
    date = datetime.today().strftime('%y%m%d')
    if prefix is None:
        prefix = 'ollam'
    if name_id is None:
        name_id = hash(name_id)
    if part_no is None:
        rand_int = os.urandom(42)
        part_no = np.base_repr(abs(hash(rand_int)), 32)
    else:
        part_no = f"{part_no:03d}" if isinstance(part_no, int) else f"{part_no}"
    key = np.base_repr(abs(hash((f'{prefix}_{date}', name_id))), 32)
    fname = f'{prefix}_{date}_{key}_{part_no}.{extension}'
    return fname


def load_texts(data_dir=None, data_file_extension='.txt', verbose=False):
    """
    Load data text files

    Kwargs:
        data_dir <str> - path to data directory; default: ~/.ollam/data
        data_file_extension <str> - data file extension
        verbose <bool> - produce logging output
    """
    data_dir = os.path.join(ollam.DATA_DIR, 'ollam') if data_dir is None else data_dir
    ollam.utils.mkdir_p(data_dir)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.endswith(data_file_extension)]
    texts = []
    for textfile in files:
        with open(textfile, 'rb') as f:
            text = f.read().decode(encoding='utf-8')
            texts.append(text)
        if verbose:
            print(f"Loading {textfile}")
    return texts


def load_csv(csvfile, types=(int, float), verbose=False):
    """
    Load csv log files

    Args:
        csvfile <str> - path to the csv log file

    Kwargs:
        types <tuple/list> - types of column data fields
        verbose <bool> - produce logging output

    Return:
        data <dict> - json file content as dict
    """
    data = {}
    with open(csvfile, 'r') as f:
        for row in csv.DictReader(f):
            for i, key in enumerate(row):
                if key not in data:
                    data[key] = []
                typing = types[i] if i < len(types) else types[-1]
                data[key].append(typing(row[key]))
        if verbose:
            print(f"Loading {csvfile}")
    return data


def split_string(string, encoding=None):
    """
    Split string into byte array tensor

    Args:
        string <str> - split string into byte array tensor

    Kwargs:
        encoding <str> - byte/string encoding
    """
    enc = ollam.encoding if encoding is None else encoding
    return tf.strings.unicode_split(string, enc)


def join_chars(characters, encoding=None):
    """
    Join character array into string

    Args:
        characters <list(str)> - list of arbitrary characters

    Kwargs:
        encoding <str> - byte/string encoding

    Return:
        string <byte> - joined byte string
    """
    enc = ollam.encoding if encoding is None else encoding
    if isinstance(characters, str):
        return characters
    elif isinstance(characters, bytes):
        return characters.decode(enc)
    return tf.strings.join(characters)[0].numpy().decode(encoding)


def io_from_sequence(sequence):
    """
    Split text data sequence into feature and label

    Args:
        sequence <iterable> - text data sequence

    Return:
        feature_text <iterable> - feature from sequence
        label_text <iterable> - label from sequence
    """
    feature_text = sequence[:-1]
    label_text = sequence[1:]
    return feature_text, label_text


def find_whitespaces(characters, encoding=None):
    """
    Find index of whitespace in a list of characters

    Args:
        characters <list(str)> - list of arbitrary characters

    Kwargs:
        encoding <str> - byte/string encoding

    Return:
        index <int> - list index of the first whitespace if any
    """
    if isinstance(characters, str):
        enc = ollam.encoding if encoding is None else encoding
        characters = split_string(characters, enc)
    elif isinstance(characters, (tuple, list, np.ndarray)):
        characters = tf.constant(characters)
    matches = tf.strings.regex_full_match(characters, ' ')
    match_idcs = tf.where(matches).numpy()
    return match_idcs


def trim_to_word(characters, encoding=None):
    """
    Trim to the next complete word

    Args:
        characters <list(str)> - list of arbitrary characters

    Return:
        characters <list(str)> - list of characters trimmed to word
    """
    if isinstance(characters, str):
        enc = ollam.encoding if encoding is None else encoding
        characters = split_string(characters, enc)
    elif isinstance(characters, (tuple, list, np.ndarray)):
        characters = tf.constant(characters)
    ws_idx = find_whitespaces(characters)
    first_ws = int(ws_idx[0])
    characters = characters[first_ws+1:]
    punctuation_chars = ollam.utils.split_string(ollam.punctuation_chars)
    while characters[0] == ' ' or characters[0] in punctuation_chars:
        characters = characters[1:]
    return characters


def whitespace_pad(characters, length, encoding=None):
    """
    Pad a list of characters with whitespaces (at the beginning)

    Args:
        characters <list/tf.Tensor> - list/byte array of arbitrary characters
        length <int> - length of characters

    Return:
        characters <list(str)> - list of characters padded with whitespaces
    """
    string = join_chars(characters, encoding=encoding)
    while len(string) < length:
        string = ' ' + string
    characters = split_string(string)
    return characters


def encode_chars(characters, encoder=None, normalize=False,
                 string_encoding=None):
    """
    Encode list of characters into a sequence of numbers

    Args:
        characters <list(str)> - list of arbitrary characters

    Kwargs:
        encoder <dict> - encoder map
        normalize <bool> - normalize the encoded data

    Return:
        sequence <np.ndarray> - 
    """
    if isinstance(characters, str):
        characters = split_string(characters, encoding=string_encoding)
    elif isinstance(characters, (list, np.ndarray)):
        characters = tf.constant(characters)
    if encoder is None:
        encoder = ollam.default_encoder
    sequence = encoder(characters)
    if normalize:
        sequence = tf.cast(sequence, float) / len(encoder.get_vocabulary())
    return sequence


def excerpt_from_encoded(text_data,
                         sequence_length=None,
                         decoder=None,
                         normalized=False,
                         random_seed=None):
    """
    Random data excerpt from an encoded text

    Args:
        text_data <list/np.ndarray> - encoded text array

    Kwargs:
        decoder <dict> - decoder map
        normalized <bool> - data is normalized
        random_seed <int> - seed for RNG

    Return:
        string <str> - decoded extracted string
    """
    if sequence_length is None:
        sequence_length = ollam.sequence_length
    if decoder is None:
        decoder = ollam.default_decoder
    if random_seed is not None:
        np.random.seed(random_seed)
    rand_start = np.random.randint(0, len(text_data))
    enc_excerpt = text_data[rand_start:(rand_start+sequence_length)]
    if normalized:
        norm = len(decoder.get_vocabulary())
        excerpt_asint = tf.math.scalar_mul(norm, enc_excerpt)
        excerpt_asint = excerpt_asint + tf.convert_to_tensor(0.5)
        excerpt_asint = tf.cast(excerpt_asint, tf.int64)
        print(excerpt_asint)
    else:
        excerpt_asint = enc_excerpt
    word_chars = decoder(excerpt_asint)
    word = join_chars(word_chars)
    return word



if __name__ == "__main__":
    from tests.prototype import SequentialTestLoader
    from tests.utils_test import UtilsModuleTest
    loader = SequentialTestLoader()
    loader.proto_load(UtilsModuleTest)
    loader.run_suites()
