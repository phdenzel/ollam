"""
ollam.utils

@author: phdenzel
"""
import os
import numpy as np
from datetime import datetime
import ollam


def mkdir_p(pathname):
    """
    Create a directory as if using 'mkdir -p' on the command line

    Args:
        pathname <str> - create all directories in given path

    Return
        pathname <str> - given path returned
    """
    from errno import EEXIST
    try:
        os.makedirs(pathname)
    except OSError as exc:  # Python > 2.5
        if exc.errno == EEXIST and os.path.isdir(pathname):
            pass
        else:
            raise

def generate_filename(prefix=None, name_id=None, part_no=None, extension='h5'):
    """
    Generate a unique filename for ollam model data

    Kwargs:
        prefix <str> - prefix of file name
        name_id <int> - ID to generate unique identifier
        part_no <int> - optional part number if landmarks part of a sequence
        extension <str> - file extension; default: npy (binary numpy extension)
    """
    date = datetime.today().strftime('%y-%m-%d').replace("-", "")
    if prefix is None:
        prefix = 'ollam'
    if name_id is None:
        name_id = hash(name_id)
    if part_no is None:
        part_no = np.base_repr(abs(hash(os.urandom(42))), 32)
    else:
        part_no = f"{part_no:03d}" if isinstance(part_no, int) else f"{part_no}"
    key = np.base_repr(abs(hash((f'{prefix}_{date}', name_id))), 32)
    fname = f'{prefix}_{date}_{key}_{part_no}.{extension}'
    return fname


if __name__ == "__main__":
    filename = generate_filename(name_id=3, part_no='10')
    filename2 = generate_filename(name_id=3, part_no='12')
    print(filename, filename2)