"""
ollam.models

@author: phdenzel
"""
import os
import shutil
import json
import inspect
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard

tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ollam


class CustomModel(tf.keras.Model):
    def __init__(self, layer_specs=[], n_labels=None, feature_shape=None, name=None):
        """
        Custom tf.keras.Model mindful of states

        Kwargs:
            layer_specs <list> - layer specifications for model
            n_labels <int> - number of dimensions for input/output
            feature_shape <tuple> - tensor input shape
        """
        super(CustomModel, self).__init__(name=name)
        self.layer_specs = layer_specs
        self.n_labels = n_labels
        self.feature_shape = feature_shape
        layers, args = self.load_specs(layer_specs, n_labels, feature_shape)
        self.custom_layers = layers
        self.custom_args = args

    def get_config(self):
        config = super(__class__, self).get_config()
        config.update({"layer_specs": self.layer_specs,
                       "n_labels": self.n_labels,
                       "feature_shape": self.feature_shape,
                       "name": self.name})
        return config

    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def as_sequential(cls, layer_specs, n_labels=None, feature_shape=None):
        """
        Crate a tf.keras.layers.Sequential model
        """
        layers, args = cls.load_specs(layer_specs, n_labels, feature_shape)
        model = tf.keras.Sequential()
        model.custom_layers = layers
        model.custom_args = args
        for layer in model.custom_layers:
            model.add(layer)
        return model

    @staticmethod
    def layer_signature_args(layer_type):
        """
        Args:
            layer_type <tf.keras.layers class> - layer class
        """
        signature = inspect.signature(layer_type.__init__)
        layer_args = {}
        for param in signature.parameters.values():
            if param.name in ['self', 'args', 'kwargs']:
                continue
            if param.default is param.empty:
                layer_args[param.name] = None
        return layer_args

    @staticmethod
    def load_specs(layer_specs, n_labels=None, feature_shape=None):
        """
        Args:
            layer_specs <list> - layer specifications for model

        Kwargs:
            n_labels <int> - number of dimensions for input/output
            feature_shape <tuple> - tensor input shape

        Return:
            layers <list> - list of tf.keras.layers instances
            args <list(dict)> - layer instantiation arguments
        """
        layers = []
        args = []
        if feature_shape is not None:
            layers.append(Input(shape=feature_shape))
        for (type_str, kwargs) in layer_specs:
            layer_type = getattr(tf.keras.layers, type_str)
            layer_args = CustomModel.layer_signature_args(layer_type)
            layer_args.update(kwargs)
            for k in layer_args:
                if layer_args[k] is None:
                    layer_args[k] = n_labels
            layer = layer_type(**layer_args)
            layers.append(layer)
            args.append(layer_args)
        return layers, args

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        for i in range(len(self.custom_layers)):
            if 'return_state' in self.custom_args[i]:
                if states is None:
                    states = self.custom_layers[i].get_initial_state(x)
                x, states = self.custom_layers[i](x, initial_state=states, training=training)
            else:
                x = self.custom_layers[i](x, training=training)
        if return_state:
            return x, states
        else:
            return x


class ModelConfigurator(object):
    """
    """
    lstm3_conn = [
        ('LSTM', dict(units=64, return_sequences=True, activation='relu')),
        ('LSTM', dict(units=128, return_sequences=True, activation='relu')),
        ('LSTM', dict(units=64, return_sequences=False, activation='relu')),
        ('Dense', dict(units=64, activation='relu')),
        # ('Dense', dict(units=32, activation='relu')),
        ('Dense', dict(activation='softmax')),
    ]
    gru_conn = [
        ('Embedding', dict(output_dim=256)),
        ('GRU', dict(units=1024, return_sequences=True, return_state=True)),
        ('Dense', dict(activation='linear')),
    ]

    def __init__(self, dataset=None, label_set=None,
                 onehot_encode=False,
                 sequence_length=None,
                 save_extension='',
                 random_state=42,
                 verbose=False):
        """
        Kwargs:
            dataset <tuple/list/tf.data.Dataset> - data arrays, 
                                                   e.g. (features, labels)
            label_set <list> - sorted set of unique labels
            onehot_encode <bool> - one-hot encode label data
            sequence_length <int> - length of dataset sequences
            train_split <float> - perform automatic test-train split
            save_extension <str> - extension of model filenames
            random_state <int> - random seed
            verbose <bool> - produce logging output
        """
        self.load_data(dataset, label_set=label_set,
                       onehot_encode=onehot_encode)
        self.sequence_length = sequence_length
        self.save_extension = save_extension
        if save_extension and not save_extension.startswith('.'):
            self.save_extension = '.' + self.save_extension
        self.random_state = random_state
        self.configure()

    @classmethod
    def from_json(cls, json_file, generate=True, dataset=None,
                  onehot_encode=True, verbose=False):
        """
        Args:
            json_file <str> - json file containing model configurations

        Kwargs:
            generate <bool> - automatically generate model using json configs
            data <tuple/list> - dataset arrays, e.g. (features, labels)
            train_split <bool> - perform automatic test-train split
            onehot_encode <bool> - one-hot encode label data
            save_extension <str> - extension of model filenames
            random_state <int> - random seed
            verbose <bool> - produce logging output

        Return:
            <ModelConfigurator instance>
        """
        with open(json_file) as f:
            kwargs = json.load(f)
            if verbose:
                print("Loading ", json_file)
        self = cls(dataset=dataset,
                   onehot_encode=onehot_encode,
                   train_split=train_split,
                   verbose=verbose)
        if generate:
            self.configure(**kwargs).generate()
        else:
            self.configure(**kwargs)
        return self

    @staticmethod
    def encode_text(text, sequence_length, normalize=False):
        """
        Args:
            texts <list(str)> - text file contents
            sequence_length <int> - encoding sequence for labelling

        Kwargs:
            normalize <bool> - normalize feature data

        Return:
            features <list/np.ndarray> - feature data array
            labels <list/np.ndarray> - label data array
            label_map <dict> - label encoding information;
                               labels (str) -> index (int)
        """
        label_set = sorted(set(text))
        label_map = tf.keras.layers.StringLookup(
            vocabulary=label_set, mask_token=None)
        text_data = ollam.utils.encode_chars(text, encoder=label_map,
                                             normalize=normalize)
        dataset = tf.data.Dataset.from_tensor_slices(text_data)
        sequences = dataset.batch(sequence_length+1, drop_remainder=True)
        dataset = sequences.map(ollam.utils.io_from_sequence)
        return dataset, label_set

    @classmethod
    def from_text(cls, text, sequence_length=None, normalize=False, **kwargs):
        """
        Args:
            text <str> - text file content

        Kwargs:
            sequence_length <int> - encoding sequence for labelling
            onehot_encode <bool> - one-hot encode label data
            normalize <bool> - normalize the data
            save_extension <str> - extension of model filenames
            random_state <int> - random seed
            verbose <bool> - produce logging output
            
        Return:
            <ModelConfigurator instance>
        """
        if isinstance(text, (tuple, list)):
            text = "\n\n\n".join(text)
        sequence_length = ollam.sequence_length if sequence_length is None else sequence_length
        dataset, label_set = cls.encode_text(text, sequence_length, normalize=normalize)
        self = cls(dataset=dataset,
                   label_set=label_set,
                   sequence_length=sequence_length,
                   **kwargs)
        return self

    @classmethod
    def from_archive(cls, model_name=None, from_checkpoints=False,
                     verbose=False, **kwargs):
        """
        Kwargs:
            model_name <str> - the archived model's name ID
            from_checkpoints <bool> - load model from checkpoint file
            verbose <bool> - produce logging output
            **kwargs <**dict> - additional keywords for 'from_json' class method

        Return:
            <ModelConfigurator instance>
        """
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
        self.load_model(from_checkpoints=from_checkpoints, verbose=verbose)
        return self

    def roll_basename(self, directory=None, randomize=True):
        """
        Generate unique basename path

        Kwargs:
            directory <str> - directory path to the basename
            randomize <bool> - randomize name ID

        Return:
            basename <str> - unique basename path
        """
        directory = ollam.MDL_DIR if directory is None else directory
        name_id = os.urandom(self.random_state) if randomize else None
        basename = os.path.join(directory, '_'.join(
            ollam.utils.generate_filename(name_id=name_id,
                                          extension='').split('_')[:-1]))
        return basename

    def load_data(self, dataset, label_set, onehot_encode=False):
        """
        Args:
            dataset <tuple(np.ndarray)/tf.data.Dataset> - dataset input
            label_set <set/list> - set of unique labels

        Kwargs:
            train_split <float> - perform automatic test-train split
            onehot_encode <bool> - one-hot encode label data
        """
        if dataset is None:
            self.dataset = dataset
            self.label_set = label_set
            self.onehot_encode = onehot_encode
            return dataset, label_set
        if not isinstance(dataset, tf.data.Dataset):
            dataset = tf.data.Dataset.from_tensor_slices(tuple(dataset))
        features = np.array(tuple(zip(*dataset))[0])
        labels = np.array(tuple(zip(*dataset))[1])
        if label_set is None:
            label_set = sorted(set(labels))
        if onehot_encode:
            labels = tf.one_hot(labels, len(label_set)).numpy()
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        self.dataset = dataset
        self.label_set = label_set
        self.onehot_encode = onehot_encode
        return dataset, label_set

    @property
    def label_map(self):
        """
        Label map to map labels to numbers (for preprocessing)

        Return:
            label_map <dict> - label encoding information;
                               labels (str) -> index (int)

        Note:
            lazy-loading
        """
        if not hasattr(self, '_label_map'):
            self._label_map = tf.keras.layers.StringLookup(
                vocabulary=self.label_set, mask_token=None)
        return self._label_map

    @label_map.setter
    def label_map(self, label_map):
        """
        Setter for label map

        Args:
            label_map <dict> - label encoding information;
                               labels (str) -> index (int)
        """
        self._label_map = label_map
    
    @property
    def inference_map(self):
        """
        Reversed label map to map numbers to labels (for inference)

        Return:
            inference_map <dict> - label decoding information;
                                   index (int) -> labels (str)

        Note:
            lazy-loading
        """
        if not hasattr(self, '_inference_map'):
            self._inference_map = tf.keras.layers.StringLookup(
                vocabulary=self.label_map.get_vocabulary(),
                invert=True, mask_token=None)
        return self._inference_map

    @inference_map.setter
    def inference_map(self, inference_map):
        """
        Setter for inference map

        Args:
            inference_map <dict> - label decoding information;
                                   index (int) -> labels (str)
        """
        self._inference_map = inference_map

    @property
    def features(self):
        if self.dataset is not None:
            return np.array(tuple(zip(*self.dataset))[0])

    @property
    def labels(self):
        if self.dataset is not None:
            return np.array(tuple(zip(*self.dataset))[1])

    @property
    def data_batch(self):
        """
        Batch and prefetch dataset
        """
        dataset = (self.dataset
                   .shuffle(self.buffer_size)
                   .batch(self.batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        return dataset

    @property
    def X(self):
        return np.array(tuple(zip(*self.data_batch))[0])

    @property
    def y(self):
        return np.array(tuple(zip(*self.data_batch))[1])

    @property
    def dataset_train(self):
        if self.train_split is not None:
            n_train = int(self.train_split * len(self.data_batch))
            return self.data_batch.take(n_train)
        return self.data_batch

    @property
    def X_train(self):
        return np.array(tuple(zip(*self.dataset_train))[0])

    @property
    def y_train(self):
        return np.array(tuple(zip(*self.dataset_train))[1])

    @property
    def dataset_test(self):
        if self.train_split is not None:
            n_train = int(self.train_split * len(self.data_batch))
            n_test = len(self.dataset) - n_train
            return self.data_batch.skip(n_train).take(n_test)
        return self.data_batch

    @property
    def X_test(self):
        return np.array(tuple(zip(*self.dataset_test))[0])

    @property
    def y_test(self):
        return np.array(tuple(zip(*self.dataset_test))[1])

    def configure(self, **kwargs):
        """
        Configuration setter

        Kwargs:
            basename <str> - unique basename path
            label_set <list> - sorted set of unique labels
            onehot_encode <bool> - one-hot encode label data
            save_extension <str> - extension of model filenames
            layer_specs <list(tuple)> - layer specifications;
                                        format: (<layer_type>, <nodes>, <activation>)
            optimizer <str> - appropriate optimizer function name
            learning_rate <float> - optimizer learning rate
            loss <str> - appropriate loss function name
            from_logits <bool> - model output from logits (non-normalized output)
            metrics <list(str)> - appropriate metrics
            train_split <float> - perform automatic test-train split
            epochs <int> - number of epoch cycles
            buffer_size <int> - shuffling buffer size of the dataset
            batch_size <int> - number of batches for training
            validation_split <float> - data ratio for monitoring trainings
            validation_data <tuple(np.ndarray)> - separate data for monitoring trainings
            validation <dict> - validation results
        """
        for (conf, dval) in [
                ('basename', self.roll_basename()),
                ('label_set', None),
                ('onehot_encode', False),
                ('save_extension', ''),
                ('train_split', None),
                ('layer_specs', self.gru_conn),
                ('optimizer', 'Adam'),
                ('learning_rate', 1e-3),
                ('loss', 'CategoricalCrossentropy'),
                ('from_logits', False),
                ('metrics', ['categorical_accuracy']),
                ('epochs', 100),
                ('buffer_size', 10000),
                ('batch_size', None),
                ('validation_split', 0.1),
                ('validation', {'loss': None}),]:
            default = dval if not hasattr(self, conf) else self.__getattribute__(conf)
            setattr(self, conf, kwargs.get(conf, default))
        return self

    @property
    def configs(self):
        return {
            'basename': self.basename,
            'label_set': self.label_set,
            'onehot_encode': self.onehot_encode,
            'save_extension': self.save_extension,
            'train_split': self.train_split,
            'layer_specs': self.layer_specs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'from_logits': self.from_logits,
            'metrics': self.metrics,
            'epochs': self.epochs,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'validation': self.validation,
        }

    @property
    def optimizer_func(self):
        return getattr(tf.keras.optimizers, self.optimizer)(
            learning_rate=self.learning_rate)

    @property
    def loss_func(self):
        return getattr(tf.losses, self.loss)(
            from_logits=self.from_logits)

    @property
    def metrics_funcs(self):
        return [getattr(tf.metrics, m)() for m in self.metrics]

    def sequential(self, **kwargs):
        """
        Simple, generic generator for Sequential models

        Kwargs:
            layer_specs <list(tuple)> - layer specifications;
                                        format: (<layer_type>, <nodes>, <activation>)
            optimizer <str> - appropriate optimizer function name
            learning_rate <float> - optimizer learning rate
            loss <str> - appropriate loss function name
            from_logits <bool> - model output from logits (non-normalized output)
            metrics <list(str)> - appropriate metrics
            train_split <float> - perform automatic test-train split
            epochs <int> - number of epoch cycles
            buffer_size <int> - shuffling buffer size of the dataset
            batch_size <int> - number of batches for training
            validation_split <float> - data ratio for monitoring trainings
            validation_data <tuple(np.ndarray)> - separate data for monitoring trainings
            verbose <bool> - print information to stdout

        Return:
            model <tf.keras.Sequential>
        """
        verbose = kwargs.pop('verbose', False)
        self.configure(**kwargs)
        n_layers = len(self.layer_specs)
        n_labels = len(self.label_set)
        input_shape = self.dataset.element_spec[0].shape + (1,)
        output_shape = self.dataset.element_spec[1].shape
        model = CustomModel.as_sequential(self.layer_specs,
                                          n_labels=n_labels,
                                          feature_shape=input_shape)
        opt = self.optimizer_func
        loss = self.loss_func
        metrics = self.metrics_funcs
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        if verbose:
            model.summary()
        return model

    def custom(self, **kwargs):
        """
        Generator for CustomModel models

        Kwargs:
            layer_specs <list(tuple)> - layer specifications;
                                        format: (<layer_type>, <nodes>, <activation>)
            optimizer <str> - appropriate optimizer function name
            learning_rate <float> - optimizer learning rate
            loss <str> - appropriate loss function name
            from_logits <bool> - model output from logits (non-normalized output)
            metrics <list(str)> - appropriate metrics
            train_split <float> - perform automatic test-train split
            epochs <int> - number of epoch cycles
            buffer_size <int> - shuffling buffer size of the dataset
            batch_size <int> - number of batches for training
            validation_split <float> - data ratio for monitoring trainings
            validation_data <tuple(np.ndarray)> - separate data for monitoring trainings
            verbose <bool> - print information to stdout

        Return:
            model <ollam.models.CustomModel>
        """
        verbose = kwargs.pop('verbose', False)
        self.configure(**kwargs)
        n_layers = len(self.layer_specs)
        n_labels = len(self.label_map.get_vocabulary())
        input_shape = self.dataset.element_spec[0].shape
        output_shape = self.dataset.element_spec[1].shape
        model = CustomModel(self.layer_specs,
                            n_labels=n_labels,
                            feature_shape=None)
        opt = self.optimizer_func
        loss = self.loss_func
        metrics = self.metrics_funcs
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        if verbose:
            self.model_summary(verbose=verbose)
        return model

    def generate(self, gen_type='custom', **kwargs):
        """
        Generate a Tensorflow model with the current configurations

        Kwargs:
            layer_specs <list(tuple)> - layer specifications;
                                        format: (<layer_type>, <nodes>, <activation>)
            optimizer <str> - appropriate optimizer function name
            learning_rate <float> - optimizer learning rate
            loss <str> - appropriate loss function name
            metrics <list(str)> - appropriate metrics
            train_split <float> - perform automatic test-train split
            epochs <int> - number of epoch cycles
            buffer_size <int> - shuffling buffer size of the dataset
            batch_size <int> - number of batches for training
            validation_split <float> - data ratio for monitoring trainings
            validation_data <tuple(np.ndarray)> - separate data for monitoring trainings
            verbose <bool> - print information to stdout

        Return:
            model <tf.keras.Sequential>
        """
        if gen_type == 'custom':
            self.model = self.custom(**kwargs)
        elif gen_type == 'sequential':
            self.model = self.sequential(**kwargs)
        return self.model

    def model_summary(self, verbose=True):
        """
        Print model summary to stdout
        """
        # generate a model if not already done
        if not hasattr(self, 'model'):
            self.generate()
        # run on test batch
        dataset = self.dataset_train

        for input_take, target_take in dataset.take(1):
            example_predictions = self.model(input_take)
            shape_str = "# (batch_size, sequence_length, label_set_size)"
            if len(example_predictions.shape) == 2:
                shape_str = "# (sequence_length, label_set_size)"
            if verbose > 1:
                print(example_predictions.shape, shape_str)
        if verbose:
            self.model.summary()

    def train_model(self, initial_epoch=0,
                    callbacks=[],
                    checkpoint_callback=False, earlystop_callback=False,
                    csv_callback=True, tensor_board_callback=False,
                    make_plots=True, archive=True, save=False, verbose=False,
                    **kwargs):
        """
        Train the generated model using optional pre-configured callbacks

        Kwargs:
            train_split <float> - perform automatic test-train split
            epochs <int> - number of epoch cycles
            buffer_size <int> - shuffling buffer size of the dataset
            batch_size <int> - number of batches for training
            validation_split <float> - data ratio for monitoring trainings

            callbacks <list> - external callback functions
            checkpoint_callback <bool> - use pre-configured tf.keras.callbacks.ModelCheckpoint
            earlystop_callback <bool> - use pre-configured tf.keras.callbacks.EarlyStopping
            tensor_board_callback <bool> - use pre-configured tf.keras.callbacks.TensorBoard
            make_plots <bool> - create training performace plots
            archive <bool> - archive model and its configurations in ollam.MDL_DIR
            save <bool> - save model and its configurations
            verbose <bool> - print information to stdout
        """
        # configure from input
        self.configure(**kwargs)
        log_dir = ollam.LOG_DIR
        ollam.utils.mkdir_p(log_dir)
        mdl_dir = ollam.MDL_DIR  # os.path.dirname(self.basename)
        ollam.utils.mkdir_p(mdl_dir)

        # set up training dataset batches
        dataset = self.dataset_train

        # callbacks
        if checkpoint_callback:
            ckpt_marker = '_cp{}'.format(self.save_extension)
            ckpt_path = self.basename + '_model{epoch:04d}'
            if not self.save_extension.endswith('h5'):
                ckpt_path = os.path.join(ckpt_path+'_cp', 'model{epoch:04d}')
            ckpt_path = ckpt_path + ckpt_marker
            save_weights_only = True if not self.save_extension.endswith('h5') else False
            cp_callback = ModelCheckpoint(ckpt_path,
                                          save_weights_only=save_weights_only,
                                          monitor='val_'+ollam.utils.camel_to_snake(self.metrics[0]),
                                          mode='max',
                                          save_best_only=True,
                                          )
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
        training_specs = {
            'epochs': self.epochs,
            'initial_epoch': initial_epoch,
            'callbacks': callbacks}
        if isinstance(dataset, (np.ndarray, tf.Tensor)):
            training_specs['validation_split'] = self.validation_split
        elif self.validation_split:
            n_data = len(dataset)
            n_val = int(self.validation_split * n_data)
            n_train = n_data - n_val
            dataset_val = dataset.skip(n_train).take(n_val)
            training_specs['validation_data'] = dataset_val
            dataset = dataset.take(n_train)
        history = self.model.fit(dataset, **training_specs)
        self.history = history.history
        
        if make_plots:
            self.performance_plots(savename=self.basename+'_tperf_{}.pdf')
        # archive and save model+log files
        if archive:
            self.basename = self.archive_model(model_dir=mdl_dir, log_dir=log_dir)
        if save:
            self.save(verbose=verbose)
        return self.history
    
    @staticmethod
    def training_performance_plots(history, savename=None, metrics=['accuracy'],
                                   save=False):
        """
        Plot the training performace history

        Args:
            history <dict> - training history of monitored metrics
            savename <str> - path to the plot file
            metrics <list(str)> - specific metrics to plot
            save <bool> - save figure

        Return:
            fig <matplotlib.figure.Figure>
            ax <matplotlib.axes.Axes>
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
        ax.set_ylabel(metrics[0])
        ax.set_xlabel("epochs")
        if save:
            plt.savefig(savename)
        return fig, ax

    def performance_plots(self, savename=None, save=True):
        """
        Plot accuracy and loss training performace

        Kwargs:
            savename <str> - path to the plot file;
                             use formatting braces to distinguish between accuracy and loss
            save <bool> - save figures
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

    def archive_model(self, model_dir=None, log_dir=None, intermediate_checkpoints=True):
        """
        Archive the current model files (move them into their own directory)

        Kwargs:
            model_dir <str> - model archive directory; default: ~/.ollam/models/
            log_dir <str> - log directory; default: ~/.ollam/log/
            intermediate_checkpoints <bool> - archive all intermediate model checkpoints

        Return:
            basename <str> - unique basename path
        """
        model_dir = os.path.dirname(self.basename) if model_dir is None else model_dir
        log_dir = ollam.LOG_DIR if log_dir is None else log_dir
        mdl_name = self.basename.split('_')[2]
        archive_dir = os.path.join(model_dir, mdl_name)
        # move models into archive directory
        ollam.utils.mkdir_p(archive_dir)
        mdl_files = [f for f in os.listdir(model_dir)
                     if os.path.isfile(os.path.join(model_dir, f))
                     or f.endswith('cp')]
        for f in mdl_files:
            fsrc = os.path.join(model_dir, f)
            fdst = os.path.join(archive_dir, f)
            os.rename(fsrc, fdst)
        # handle checkpoint files
        basename = os.path.join(archive_dir, os.path.basename(self.basename))
        ckpt_marker = 'cp{}'.format(self.save_extension)
        ckpt_path = basename + '_model_' + ckpt_marker
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)
        cp_files = sorted([fmdl for fmdl in os.listdir(archive_dir)
                           if fmdl.endswith(ckpt_marker)])
        if not intermediate_checkpoints:
            for f in cp_files[:-1]:
                shutil.rmtree(os.path.join(archive_dir, f))
        for f in cp_files[-1:]:
            shutil.copytree(os.path.join(archive_dir, f), ckpt_path)
        # archive log directory
        mdl_log_dir = os.path.join(archive_dir, os.path.basename(log_dir))
        shutil.copytree(log_dir, mdl_log_dir, dirs_exist_ok=True)
        shutil.rmtree(log_dir)
        return basename

    def save(self, weights_only=False, directory=None, overwrite=True, verbose=False):
        """
        Save configs and model

        Kwargs:
            directory <str> - directory where the files are saved
            overwrite <bool> - overwrite if files already exist
            verbose <bool> - print information to stdout
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
            modelfile = self.basename+'_model{}'.format(self.save_extension)
            if modelfile.endswith('h5'):
                self.model.save(modelfile)
            else:
                self.model.save_weights(modelfile)
            if verbose:
                print(f"Writing to {modelfile}")

    def load_model(self, save_extension=None, from_checkpoints=False, verbose=False):
        """
        Load model from either last model state or checkpoint

        Kwargs:
            from_checkpoints <bool> - load the checkpoint saves instead
            verbose <bool> - print information to stdout
        """
        if save_extension is not None:
            self.save_extension = save_extension
        model_filename = self.basename+'_model{}'.format(self.save_extension)
        cp_filename = self.basename+'_model_cp{}'.format(self.save_extension)
        # print(model_filename)
        # print(cp_filename)
        if not hasattr(self, 'model'):
            self.generate()
        if from_checkpoints and os.path.exists(cp_filename):
            self.model.load_weights(cp_filename)
            if verbose:
                print("Loading", cp_filename)
        elif os.path.isfile(model_filename):
            if not hasattr(self, 'model'):
                self.generate()
            self.model = tf.keras.models.load_model(model_filename)
            if verbose:
                print("Loading", model_filename)
        history_filename = self.basename+'_history.log'
        if os.path.isfile(history_filename):
            self.history = ollam.utils.load_csv(history_filename,
                                                verbose=verbose)

    def validate(self, verbose=False):
        """
        Run a validation using the current model

        Kwargs:
            verbose <bool> - print information to stdout
        """
        # set up training dataset batches
        dataset = self.dataset_test
        for test_feature, test_label in dataset.take(1):
            # inference
            yhat = self.model(test_feature, return_state=False, training=False)
            # loss and metrics
            batch_loss = self.loss_func(test_label, yhat)
            batch_metrics = [f(test_label, yhat) for f in self.metrics_funcs]
            mean_loss = batch_loss.numpy().mean().astype(float)
            mean_metrics = [b.numpy().mean().astype(float) for b in batch_metrics]
            self.validation['loss'] = mean_loss
            self.validation['metrics'] = mean_metrics
            if verbose:
                print("Mean of loss   \t", mean_loss)
                print("Mean of metrics\t", mean_metrics)


def train(data_dir=None, return_obj=True, **kwargs):
    """
    Generate and train a default lstm3_conn model with the ModelConfigurator
    """
    import pprint
    data_dir = ollam.DATA_DIR if data_dir is None else data_dir
    onehot_encode = kwargs.pop('onehot_encode', False)
    sequence_length = kwargs.pop('sequence_length', ollam.sequence_length)
    random_state = kwargs.pop('random_state', 8)
    verbose = kwargs.pop('verbose', 2)

    model_type = 'custom'
    kwargs.setdefault('layer_specs', ModelConfigurator.gru_conn)
    kwargs.setdefault('optimizer', ollam.optimizer)
    kwargs.setdefault('learning_rate', ollam.learning_rate)
    kwargs.setdefault('loss', 'SparseCategoricalCrossentropy')
    kwargs.setdefault('metrics', ['SparseCategoricalAccuracy'])
    kwargs.setdefault('from_logits', True)
    kwargs.setdefault('epochs', ollam.epochs)
    kwargs.setdefault('train_split', 0.95)
    kwargs.setdefault('batch_size', ollam.batch_size)
    kwargs.setdefault('validation_split', ollam.validation_split)

    if verbose:
        print(f"### ModelConfigurator: {model_type}")

    texts = ollam.utils.load_texts(data_dir, verbose=verbose)
    configurator = ModelConfigurator.from_text(
        texts, sequence_length=sequence_length,
        onehot_encode=onehot_encode, normalize=False)
    configurator.configure(**kwargs)
    if verbose:
        print("### Configs:")
        pprint.pprint(configurator.configs)
    configurator.generate(model_type, verbose=verbose)

    if verbose:
        print("### Testing model on data")
        configurator.validate(verbose=verbose)

    configurator.train_model(checkpoint_callback=True,
                             tensor_board_callback=False,
                             archive=True,
                             save=True,
                             verbose=verbose)

    if verbose:
        print(f"# Validation (Epoch {configurator.epochs})")
    configurator.validate(verbose=verbose)

    if verbose:
        print("# Validation (Checkpoint)")
    configurator.load_model(from_checkpoints=True, verbose=verbose)
    configurator.validate(verbose=verbose)
    # validation_cp = configurator.validation.copy()
    # configurator.save(overwrite=False)
    # if return_obj:
    #     return configurator
    # else:
    #     del configurator
    #     tf.keras.backend.clear_session()
    #     return validation_cp, validation_end    


if __name__ == "__main__":
    train()

    
    # continue_train('6C7LES4K13HL9', epochs=200, initial_epoch=150)

    # configurator = load_from_archive('4BQ0JB4TEA2E7')
    # configurator.performance_plots(save=False)
    # plt.show()

    process('6C7LES4K13HL9', 200, from_checkpoints=True)
    # process('6C7LES4K13HL9', 400, init='in the forest he', from_checkpoints=True)
