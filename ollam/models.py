"""
ollam.models

@author: phdenzel
"""
import os
import shutil
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard

tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ollam


class ModelConfigurator(object):
    """
    """
    lstm3_conn = [('LSTM', 128, 'relu'),
                  # ('Dropout', 0.2, None),
                  ('LSTM', 256, 'relu'),
                  # ('Dropout', 0.2, None),
                  ('LSTM', 128, 'relu'),
                  # ('Dropout', 0.2, None),
                  ('Dense', 128, 'relu'),
                  ('Dense', 64, 'relu'),
                  ('Dense', 0, 'softmax')]

    def __init__(self, features=None, labels=None,
                 label_map=None,
                 train_split=False,
                 categorical=True,
                 random_state=42):
        """
        Kwargs:
            features <list/np.ndarray> -
            labels <list/np.ndarray> -
            train_split <bool> -
            categorical <bool> -
            random_state <int> -
        """
        self.categorical = categorical
        self.random_state = random_state
        self.label_map = label_map
        self.load_data(features, labels, train_split=train_split)
        self.configure()

    @classmethod
    def from_json(cls, json_file,
                  features=None, labels=None,
                  generate=True, train_split=False,
                  categorical=True, random_state=42,
                  verbose=False):
        """
        """
        with open(json_file) as f:
            kwargs = json.load(f)
            if verbose:
                print("Loading ", json_file)
        self = cls(features=features, labels=labels,
                   random_state=random_state)
        if generate:
            self.configure(**kwargs).generate()
        else:
            self.configure(**kwargs)
        return self

    @staticmethod
    def encode_text(texts, sequence_length):
        texts = [text.lower() for text in texts]
        alphabet = sorted(list(set("".join(texts))))
        label_map = {char: n for n, char in enumerate(alphabet)}
        labels = []
        features = []
        for text in texts:
            for i in range(len(text) - sequence_length):
                sequence = text[i:i+sequence_length]
                follow = text[i+sequence_length]
                feature = [label_map[char] for char in sequence]
                features.append(feature)
                labels.append(follow)
        features = np.asarray(features).reshape((len(features), sequence_length, 1))
        labels = np.asarray(labels)
        return features, labels, label_map

    @classmethod
    def from_text(cls, text, sequence_length=None, train_split=False,
                  categorical=True, random_state=42):
        sequence_length = ollam.sequence_length if sequence_length is None else sequence_length
        features, labels, label_map = cls.encode_text([text], sequence_length)
        self = cls(features=features, labels=labels, label_map=label_map,
                   categorical=categorical, train_split=False,
                   random_state=random_state)
        return self

    @classmethod
    def from_texts(cls, texts, sequence_length=None, train_split=False,
                   categorical=True, random_state=42):
        sequence_length = ollam.sequence_length if sequence_length is None else sequence_length
        features, labels, label_map = cls.encode_text(texts, sequence_length)
        self = cls(features=features, labels=labels, label_map=label_map,
                   categorical=categorical, train_split=False,
                   random_state=random_state)
        return self

    @classmethod
    def from_archive(cls, model_name=None, from_checkpoints=False,
                     verbose=False, **kwargs):
        mdl_dir = ollam.MDL_DIR
        if model_name is None:
            models = os.listdir(mdl_dir)
            models = models if len(models) > 0 else [None]
            model_name = 'ollam' if 'ollam' in models else models[-1]
        if model_name is None:
            print("Please first build and train a model!")
        model_path = os.path.join(mdl_dir, model_name)
        jsn = [f for f in os.listdir(model_path) if f.endswith('json')][0]
        jsn_path = os.path.join(model_path, jsn)
        self = cls.from_json(jsn_path, generate=False, verbose=verbose, **kwargs)
        self.load_model(from_checkpoints=from_checkpoints,
                        verbose=verbose)
        return self

    def load_data(self, features, labels, categorical=None,
                  train_split=False):
        """
        Args:
            features <list/np.ndarray> -
            labels <list/np.ndarray> -

        Kwargs:
            categorical <bool> -
        """
        if categorical is not None:
            self.categorical = categorical
        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        self.label_set = set(self.labels) if np.any(self.labels.shape) else []
        if self.label_map is None:
            self.label_map = self.create_label_map(
                self.labels, label_set=self.label_set)
        if train_split:
            self.test_train_split()
        
    @staticmethod
    def create_label_map(labels, label_set=None):
        """
        Map labels to numbers for modelling
        
        Args:
            labels <list> - a list of all labels

        Return:
            label_map <dict> - labels (str) -> index (int)
        """
        labels = np.asarray(labels)
        labels = [None] if not np.any(labels.shape) else labels
        unique_labels = sorted(set(labels)) if label_set is None else label_set
        label_map = {}
        for idx, label in enumerate(unique_labels):
            label_map[label] = idx
        return label_map
    
    @property
    def inv_label_map(self):
        """
        Reversed label map to map numbers to labels (for inference)

        Return:
            inv_label_map <dict> - index (int) -> labels (str)
        """
        if self.label_map is not None:
            return {self.label_map[k]: k for k in self.label_map}

    def roll_basename(self, directory=None, randomize=True):
        """
        Kwargs:
            directory <str> -
        """
        directory = ollam.MDL_DIR if directory is None else directory
        name_id = os.urandom(self.random_state) if randomize else None
        basename = os.path.join(directory, '_'.join(
            ollam.utils.generate_filename(name_id=name_id,
                                          extension='').split('_')[:-1]))
        return basename

    def configure(self, **kwargs):
        """
        Kwargs:
            basename <str> -
            layer_specs <list(tuple)> -
            optimizer <str> -
            learning_rate <float> -
            loss <str> -
            metrics <list(str)> -
            epochs <int> -
            batch_size <int> -
            validation_split <float> -
            validation_data <tuple(np.ndarray)> -
            validation <dict> -
        """
        for (conf, dval) in [('layer_specs', self.lstm3_conn), ('optimizer', 'Adam'),
                             ('learning_rate', 1e-3), ('loss', 'categorical_crossentropy'),
                             ('metrics', ['categorical_accuracy']), ('epochs', 1000),
                             ('batch_size', None), ('validation_split', 0.2),
                             ('validation_data', None),
                             ('validation', {'accuracy': None, 'confm': None}),
                             ('basename', self.roll_basename()),
                             ('label_map', self.create_label_map(self.labels))]:
            default = dval if not hasattr(self, conf) else self.__getattribute__(conf)
            setattr(self, conf, kwargs.get(conf, default))
        return self

    @property
    def configs(self):
        return {
            'basename': self.basename,
            'label_map': self.label_map,
            'layer_specs': self.layer_specs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'metrics': self.metrics,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'validation_data': self.validation_data,
            'validation': self.validation,
        }

    @property
    def labels_as_int(self):
        if np.any(self.labels.shape):
            return np.array([self.label_map[l] for l in self.labels])
        return np.array(self.labels)

    @property
    def X(self):
        return self.features

    @property
    def y(self):
        y = self.labels_as_int
        if self.categorical:
            y = to_categorical(y).astype(int)
        return y

    @property
    def X_train(self):
        if not hasattr(self, 'train_data'):
            return self.X
        return self.train_data[0]

    @property
    def y_train(self):
        if not hasattr(self, 'train_data'):
            return self.y
        return self.train_data[1]

    @property
    def X_test(self):
        if not hasattr(self, 'test_data'):
            return self.X
        return self.test_data[0]

    @property
    def y_test(self):
        if not hasattr(self, 'test_data'):
            return self.y
        return self.test_data[1]

    def test_train_split(self, test_size=0.1, random_state=None):
        """
        Kwargs:
            test_size <float> -
            random_state <int> -
        """
        if random_state is None:
            random_state = self.random_state
        if np.any(self.X.shape) and np.any(self.y.shape):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)
            self.train_data = X_train, y_train
            self.test_data = X_test, y_test

    def sequential(self, **kwargs):
        """
        Simple, generic generator for Sequential models

        Kwargs:
            layer_specs <list> - tuples of (layer_type, nodes, activation)
            optimizer <str> - appropriate optimizer function name
            loss <str> - appropriate loss function name
            metrics <list(str)> - appropriate metrics
            verbose <bool> - print information to stdout

        Return:
            model <tf.keras.models.Sequential>
        """
        verbose = kwargs.pop('verbose', False)
        self.configure(**kwargs)
        model = Sequential()
        n_layers = len(self.layer_specs)
        layer_types = [l[0] for l in self.layer_specs]
        input_shape = self.X.shape[1:]
        output_shape = self.y.shape[-1]
        for i, (layer_type_str, nodes, activation) in enumerate(self.layer_specs):
            layer_type = getattr(tf.keras.layers, layer_type_str)
            if i == 0:
                model.add(Input(shape=input_shape))
            args = (output_shape,) if i == n_layers-1 else (nodes,)
            kw = dict(activation=activation)
            if layer_type_str == 'LSTM':
                kw['return_sequences'] = True
            if layer_type_str == 'Dropout':
                kw.pop('activation')
            if i == n_layers-layer_types[::-1].index('LSTM')-1:
                kw['return_sequences'] = False
            model.add(layer_type(*args, **kw))
        opt = getattr(tf.keras.optimizers, self.optimizer)(
            learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        if verbose:
            model.summary()
        return model

    def generate(self, **kwargs):
        """
        Generate a Tensorflow model with the current configurations

        Kwargs:
            layer_specs <list> - tuples of (layer_type, nodes, activation)
            optimizer <str> - appropriate optimizer function name
            loss <str> - appropriate loss function name
            metrics <list(str)> - appropriate metrics
            verbose <bool> - print information to stdout

        Return:
            model <tf.keras.models.Sequential>
        """
        self.model = self.sequential(**kwargs)
        return self.model

    def train_model(self,
                    epochs=None, batch_size=None, validation_split=None,
                    initial_epoch=0,
                    callbacks=[],
                    checkpoint_callback=False,
                    earlystop_callback=False,
                    csv_callback=True,
                    tensor_board_callback=False,
                    make_plots=True,
                    archive=True,
                    save=False,
                    verbose=False):
        """
        Kwargs:
            epochs <int> -
            batch_size <int> -
            validation_split <float> -
            callbacks <list> -
            checkpoint_callback <bool> -
            earlystop_callback <bool> -
            tensor_board_callback <bool> -
            save <bool> -
            verbose <bool> -
        """
        epochs = self.epochs if epochs is None else epochs
        self.epochs = epochs
        batch_size = self.batch_size if batch_size is None else batch_size
        self.batch_size = batch_size
        validation_split = self.validation_split if validation_split is None else validation_split
        self.validation_split = validation_split
        log_dir = ollam.LOG_DIR
        ollam.utils.mkdir_p(log_dir)
        mdl_dir = ollam.MDL_DIR  # os.path.dirname(self.basename)
        ollam.utils.mkdir_p(mdl_dir)

        # callbacks
        if checkpoint_callback:
            cp_callback = ModelCheckpoint(self.basename+'_model{epoch:04d}_cp.h5',
                                          monitor='val_'+self.metrics[0],
                                          save_best_only=True, mode='max',
                                          verbose=False)
            
            callbacks.append(cp_callback)
        if earlystop_callback:
            es_callback = EarlyStopping(monitor='val_'+self.metrics[0],
                                        min_delta=0.1, patience=25,
                                        restore_best_weights=False)
            callbacks.append(es_callback)
        if csv_callback:
            csv_callback = CSVLogger(self.basename+'_history.log', separator=',', append=True)
            callbacks.append(csv_callback)
        if tensor_board_callback:
            tb_callback = TensorBoard(log_dir=log_dir)
            callbacks.append(tb_callback)
        if not hasattr(self, 'model'):
            self.generate()

        # fit model
        history = self.model.fit(self.X_train, self.y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks)
        self.history = history.history
        
        if make_plots:
            self.performance_plots(savename=self.basename+'_tperf_{}.pdf')
        # archive model+log files and clean up
        self.basename = self.archive_model(mdl_dir=mdl_dir, log_dir=log_dir)
        if save:
            self.save(verbose=verbose)
        return self.history
    
    @staticmethod
    def training_performance_plots(history, savename=None, metrics=['accuracy'],
                                   save=False):
        """
        """
        if savename is None:
            savename = 'training_performance.pdf'
        fig, ax = plt.subplots()
        key = metrics[0]
        if key in history:
            ax.plot(history[key], c='#6767FF', label='train')
        key = 'val_'+key
        if key in history:
            ax.plot(history[key], c='#FF6767', label='test')
        ax.set_ylabel('performance')
        ax.set_xlabel("epochs")
        if save:
            plt.savefig(savename)
        return fig, ax

    def performance_plots(self, savename=None, save=True):
        """
        """
        if savename is None:
            savename = self.basename+'_tperf_{}.pdf'
        self.training_performance_plots(self.history,
                                        savename=savename.format('acc'),
                                        metrics=self.metrics,
                                        save=save)
        self.training_performance_plots(self.history,
                                        savename=savename.format('loss'),
                                        metrics=['loss'],
                                        save=save)

    def archive_model(self, mdl_dir=None, log_dir=None, intermediate_checkpoints=True):
        """
        Kwargs:
            mdl_dir <str> -
            log_dir <str> -
            intermediate_checkpoints <bool> -
        """
        mdl_dir = os.path.dirname(self.basename) if mdl_dir is None else mdl_dir
        log_dir = ollam.LOG_DIR if log_dir is None else log_dir
        # move models into archive directory
        mdl_name = self.basename.split('_')[2] #+ f'_{self.optimizer}'
        mdl_files = [f for f in os.listdir(mdl_dir)
                     if os.path.isfile(os.path.join(mdl_dir, f))]
        archive_dir = os.path.join(mdl_dir, mdl_name)
        ollam.utils.mkdir_p(archive_dir)
        for f in mdl_files:
            fsrc = os.path.join(mdl_dir, f)
            fdst = os.path.join(archive_dir, f)
            os.rename(fsrc, fdst)
        # handle checkpoint files
        basename = os.path.join(archive_dir,
                                os.path.basename(self.basename))
        if os.path.isfile(basename+'_model_cp.h5'):
            os.remove(basename+'_model_cp.h5')
        cp_files = sorted([fmdl for fmdl in os.listdir(archive_dir)
                           if fmdl.endswith('cp.h5')])
        if not intermediate_checkpoints:
            for f in cp_files[:-1]:
                os.remove(os.path.join(archive_dir, f))
        for f in cp_files[-1:]:
            shutil.copy2(os.path.join(archive_dir, f), basename+'_model_cp.h5')
        # archive log directory
        mdl_log_dir = os.path.join(archive_dir, os.path.basename(log_dir))
        shutil.copytree(log_dir, mdl_log_dir, dirs_exist_ok=True)
        shutil.rmtree(log_dir)
        return basename

    def save(self, directory=None, overwrite=True, verbose=False):
        """
        Save configs and model

        Kwargs:
            directory <str> - directory where the files are saved
        """
        if directory is not None:
            basename = os.path.join(directory, os.path.basename(self.basename))
        else:
            basename = self.basename
        jsnf = basename+'_configs.json'
        with open(jsnf, 'w') as f:
            json.dump(self.configs, f)
            if verbose:
                print(f"Writing to {jsnf}")
        if overwrite:
            modelfile = self.basename+'_model.h5'
            self.model.save(modelfile)
            if verbose:
                print(f"Writing to {modelfile}")

    def load_model(self, from_checkpoints=False, verbose=False):
        """
        Kwargs:
            from_checkpoints <bool> - load the checkpoint saves instead
        """
        model_filename = self.basename+'_model.h5'
        cp_filename = self.basename+'_model_cp.h5'
        if from_checkpoints and os.path.isfile(self.basename+'_model_cp.h5'):
            self.model = tf.keras.models.load_model(self.basename+'_model_cp.h5')
            if verbose:
                print("Loading", self.basename+'_model_cp.h5')
        elif os.path.isfile(self.basename+'_model.h5'):
            self.model = tf.keras.models.load_model(self.basename+'_model.h5')
            if verbose:
                print("Loading", self.basename+'_model.h5')
        if os.path.isfile(self.basename+'_history.log'):
            self.history = ollam.utils.load_csv(self.basename+'_history.log',
                                                verbose=verbose)

    def validate(self, verbose=False):
        """
        Kwargs:
            verbose <bool> -
        """
        y_hat = self.model.predict(self.X_test)
        y_hat = np.argmax(y_hat, axis=1).tolist()
        y_true = np.argmax(self.y_test, axis=1).tolist()
        self.validation['accuracy'] = accuracy_score(y_true, y_hat)
        self.validation['confm'] = \
            multilabel_confusion_matrix(y_true, y_hat).tolist()
        if verbose:
            print("Accuracy:        \t{}".format(self.validation['accuracy']))
            print("Confusion matrix:\t{}".format(self.validation['confm']))


def train(data_dir=None, return_obj=True, **kwargs):
    """
    Generate and train a default lstm3_conn model with the ModelConfigurator
    """
    import pprint
    data_dir = ollam.DATA_DIR if data_dir is None else data_dir
    categorical = kwargs.pop('categorical', True)
    train_split = kwargs.pop('train_split', False)
    random_state = kwargs.pop('random_state', 8)
    verbose = kwargs.pop('verbose', True)
    
    kwargs.setdefault('optimizer', ollam.optimizer)
    kwargs.setdefault('learning_rate', ollam.learning_rate)
    kwargs.setdefault('epochs', ollam.epochs)
    kwargs.setdefault('batch_size', ollam.batch_size)
    kwargs.setdefault('validation_split', ollam.validation_split)

    if verbose:
        print("# ModelConfigurator: lstm3_conn")

    texts = ollam.utils.load_text(data_dir, verbose=verbose)
    configurator = ModelConfigurator.from_text(texts[0],
                                               train_split=train_split,
                                               categorical=categorical)
    configurator.configure(**kwargs)
    if verbose:
        print("Configs:")
        pprint.pprint(configurator.configs)
    
    configurator.train_model(checkpoint_callback=True,
                             tensor_board_callback=True,
                             save=True,
                             verbose=verbose)
    if verbose:
        print(f"# Validation (Epoch {configurator.epochs})")
    configurator.validate(verbose=verbose)
    validation_end = configurator.validation.copy()
    if verbose:
        print("# Validation (Checkpoint)")
    configurator.load_model(from_checkpoints=True)
    configurator.validate(verbose=verbose)
    validation_cp = configurator.validation.copy()
    configurator.save(overwrite=False)
    if return_obj:
        return configurator
    else:
        del configurator
        tf.keras.backend.clear_session()
        return validation_cp, validation_end


def load_from_archive(mdl_dir_name, data_dir=None, **kwargs):
    """
    Load a model from previously archived json and hdf5 files
    """
    model_dir = os.path.join(ollam.MDL_DIR, mdl_dir_name)
    data_dir = ollam.DATA_DIR if data_dir is None else data_dir
    # log_dir = kwargs.pop('log_dir', ollam.LOG_DIR)
    sequence_length = kwargs.pop('sequence_length', ollam.sequence_length)
    categorical = kwargs.pop('categorical', True)
    train_split = kwargs.pop('train_split', False)
    random_state = kwargs.pop('random_state', 8)
    from_checkpoints = kwargs.pop('from_checkpoints', True)
    verbose = kwargs.pop('verbose', True)

    if verbose:
        print(f"# ModelConfigurator: load from {mdl_dir_name}")

    texts = ollam.utils.load_text(data_dir, verbose=verbose)
    features, labels, _ = ModelConfigurator.encode_text(texts, sequence_length)
    configurator = ModelConfigurator.from_archive(mdl_dir_name,
                                                  features=features, labels=labels,
                                                  categorical=categorical,
                                                  random_state=random_state,
                                                  verbose=verbose)
    return configurator


def continue_train(mdl_dir_name, epochs=100, initial_epoch=None,
                   data_dir=None, return_obj=True, **kwargs):
    """
    Continue training a model from previously archived hdf5 file
    """
    verbose = kwargs.get('verbose', True)
    configurator = load_from_archive(mdl_dir_name, from_checkpoints=False, **kwargs)
    initial_epoch = configurator.epochs if initial_epoch is None else initial_epoch
    configurator.epochs = epochs
    configurator.train_model(initial_epoch=initial_epoch,
                             checkpoint_callback=True,
                             tensor_board_callback=True,
                             save=True,
                             verbose=verbose)
    if return_obj:
        return configurator


def process(mdl_dir_name, )



if __name__ == "__main__":
    configurator = load_from_archive('S6239562ROB9')
    configurator.performance_plots(save=False)
    plt.show()
    # continue_train('S6239562ROB9', epochs=200, initial_epoch=150)
