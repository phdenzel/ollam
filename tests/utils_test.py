"""
fabular - tests._test

@author: phdenzel
"""
import os
import numpy as np
import ollam
import tensorflow as tf
from tests.prototype import UnitTestPrototype
from tests.prototype import SequentialTestLoader


class UtilsModuleTest(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_mkdir_p(self):
        """ # ollam.utils.mkdir_p """
        filename = 'tmp'
        exists = os.path.exists(filename)
        self.assertFalse(exists)
        self.printf(filename)
        ollam.utils.mkdir_p(filename)
        exists = os.path.exists(filename)
        self.assertTrue(exists)
        os.rmdir(filename)

    def test_generate_filename(self):
        """ # ollam.utils.generate_filename """
        # default kwargs
        kwargs = dict(prefix=None, name_id=None, part_no=None, extension='h5')
        self.printf(kwargs)
        filename = ollam.utils.generate_filename(**kwargs)
        self.assertIsInstance(filename, str)
        self.assertTrue(filename.startswith('ollam'))
        self.assertTrue(filename.endswith(kwargs['extension']))
        self.printout(filename)
        # with kwargs
        kwargs = dict(prefix='poet', name_id=1, part_no=42, extension='txt')
        self.printf(kwargs)
        filename = ollam.utils.generate_filename(**kwargs)
        self.assertIsInstance(filename, str)
        self.assertTrue(filename.startswith(kwargs['prefix']))
        self.assertTrue(str(kwargs['part_no']) in filename)
        self.assertTrue(filename.endswith(kwargs['extension']))
        self.printout(filename)

    def test_load_texts(self):
        """ # ollam.utils.load_texts """
        ddir = os.path.join(os.path.dirname(__file__), '../data')
        kwargs = dict(data_dir=ddir, data_file_extension='.txt', verbose=True)
        self.printf(kwargs)
        texts = ollam.utils.load_texts(**kwargs)
        self.assertIsInstance(texts, list)
        self.assertIsInstance(texts[0], str)
        self.printout(texts[0][:100])

    def test_load_csv(self):
        """ # ollam.utils.load_csv """
        csvfile = os.path.join(os.path.dirname(__file__),
                               '../data/ollam_test_history.log')
        kwargs = dict(csvfile=csvfile, types=(int, float), verbose=False)
        self.printf(kwargs)
        csv_data = ollam.utils.load_csv(**kwargs)
        self.assertIsInstance(csv_data, dict)
        self.assertTrue('epoch' in csv_data)
        for key in ['categorical_accuracy', 'loss']:
            self.assertTrue(key in csv_data)
            self.assertTrue('val_'+key in csv_data)
        self.printout(csv_data)

    def test_find_whitespace(self):
        """ # ollam.utils.find_whitespaces """
        chars = "! test characters."
        kwargs = dict(characters=chars)
        self.printf(kwargs)
        ws_idx = ollam.utils.find_whitespaces(**kwargs)
        self.assertTrue(1 in ws_idx)
        self.assertTrue(6 in ws_idx)
        self.assertIsInstance(ws_idx, np.ndarray)
        self.printout(ws_idx)
        chars = list("! test ")
        kwargs = dict(characters=chars)
        self.printf(kwargs)
        ws_idx = ollam.utils.find_whitespaces(**kwargs)
        self.assertTrue(1 in ws_idx)
        self.assertTrue(6 in ws_idx)
        self.assertIsInstance(ws_idx, np.ndarray)
        self.printout(ws_idx)
        chars = list("!test.")
        kwargs = dict(characters=chars)
        self.printf(kwargs)
        ws_idx = ollam.utils.find_whitespaces(**kwargs)
        self.assertEqual(len(ws_idx), 0)
        self.assertIsInstance(ws_idx, np.ndarray)
        self.printout(ws_idx)

    def test_split_chars(self):
        """ ollam.utils.split_string """
        string = "! test characters."
        self.printf(string)
        chars = ollam.utils.split_string(string)
        self.assertIsInstance(chars, tf.Tensor)
        for a, b in zip(chars, list(string)):
            self.assertEqual(a, b)
        self.printout(chars)

    def test_join_chars(self):
        """ # ollam.utils.join_chars """
        string = "! test characters."
        chars = list(string)
        self.printf(chars)
        joined = ollam.utils.join_chars(chars)
        self.assertIsInstance(joined, str)
        self.printout(joined)
        chars = ollam.utils.split_string(string)
        self.printf(chars)
        joined = ollam.utils.join_chars(chars)
        self.assertIsInstance(joined, str)
        self.printout(joined)

    def test_trim_to_word(self):
        """ # ollam.utils.trim_to_word """
        chars = "! test characters."
        kwargs = dict(characters=chars)
        self.printf(kwargs)
        trimmed = ollam.utils.trim_to_word(**kwargs)
        self.assertIsInstance(trimmed, tf.Tensor)
        self.assertEqual(ollam.utils.join_chars(trimmed), "test characters.")
        self.printout(trimmed)
        chars = "!  test ."
        kwargs = dict(characters=chars)
        self.printf(kwargs)
        trimmed = ollam.utils.trim_to_word(**kwargs)
        self.assertIsInstance(trimmed, tf.Tensor)
        self.assertEqual(ollam.utils.join_chars(trimmed), "test .")
        self.printout(trimmed)

    def test_whitespace_pad(self):
        """ # ollam.utils.whitespace_pad """
        chars = "test"
        kwargs = dict(characters=chars, length=8)
        self.printf(kwargs)
        chars = ollam.utils.whitespace_pad(**kwargs)
        self.assertIsInstance(chars, tf.Tensor)
        self.assertEqual(len(chars), kwargs['length'])
        self.printout(chars)
        chars = ollam.utils.split_string("test")
        kwargs = dict(characters=chars, length=6)
        self.printf(kwargs)
        chars = ollam.utils.whitespace_pad(**kwargs)
        self.assertIsInstance(chars, tf.Tensor)
        self.assertEqual(len(chars), kwargs['length'])
        self.printout(chars)

    def test_encode_chars(self):
        """ # ollam.utils.encode_chars """
        test_word = 'abcdef'
        kwargs = dict(characters=test_word, normalize=False)
        self.printf(kwargs)
        sequence = ollam.utils.encode_chars(**kwargs)
        self.assertIsInstance(sequence, tf.Tensor)
        self.printout(sequence)
        test_word = list('abcdef')
        kwargs = dict(characters=test_word, normalize=False)
        self.printf(kwargs)
        sequence = ollam.utils.encode_chars(**kwargs)
        self.assertIsInstance(sequence, tf.Tensor)
        self.printout(sequence)
        test_word = ollam.utils.split_string('abcdef')
        kwargs = dict(characters=test_word, normalize=False)
        self.printf(kwargs)
        sequence = ollam.utils.encode_chars(**kwargs)
        self.assertIsInstance(sequence, tf.Tensor)
        self.printout(sequence)
        kwargs = dict(characters=test_word, normalize=True)
        self.printf(kwargs)
        sequence = ollam.utils.encode_chars(**kwargs)
        self.assertIsInstance(sequence, tf.Tensor)
        self.printout(sequence)

    def test_excerpt_from_encoded(self):
        """ # ollam.utils.excerpt_from_encoded """
        test_word = 'abcdef'
        sequence = ollam.utils.encode_chars(test_word*3, normalize=False)
        kwargs = dict(text_data=sequence, sequence_length=4,
                      decoder=None, normalized=False, random_seed=None)
        self.printf(kwargs)
        word_chars = ollam.utils.excerpt_from_encoded(**kwargs)
        self.printout(word_chars)
        sequence = ollam.utils.encode_chars(test_word*5, normalize=True)
        kwargs = dict(text_data=sequence, sequence_length=4,
                      decoder=None, normalized=True, random_seed=None)
        self.printf(kwargs)
        word = ollam.utils.excerpt_from_encoded(**kwargs)
        self.printout(word)



if __name__ == "__main__":
    loader = SequentialTestLoader()
    loader.proto_load(UtilsModuleTest)
    loader.run_suites()
