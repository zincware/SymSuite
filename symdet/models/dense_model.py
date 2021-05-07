"""
Dense neural network model
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input



class DenseModel:
    """
    Class for the construction and training of a dense NN model

    Attributes
    ----------
    data_dict : dict
            Dictionary of data where the key is the class name and the values are the coordinates belongin to that
            class. This is fundamentall a classification problem!
    n_layers : int
            Number of hidden layers to use.
    units : int
            Number of units to use in each layer.
    epochs : int
            Number of epochs during training.
    activation : str
            Activation function to use.
    monitor : str
            Monitor parameter for use in training.
    lr : float
            Learning rate of the algorithm.
    batch_size : int
            batch size for the training.
    terminate_patience : int
            Patience value for termination to be used in a callback.
    lr_patience : int
            Patience in the learning rate reduction.
    train_ds : tf.data.Dataset
            Train dataset.
    test_ds : tf.data.Dataset
            Test dataset.
    val_ds : tf.data.Dataset
            validation dataset.
    model : tf.keras.Model
            Tensorflow model to train.
    """

    def __init__(self, data_dict: dict, n_layers: int = 5, units: int = 10, epochs: int = 20, activation: str = 'relu',
                 monitor: str = 'accuracy', lr: float = 1e-4, batch_size: int = 100, terminate_patience: int = 10,
                 lr_patience: int = 5):
        """
        Python constructor

        Parameters
        ----------
        data_dict : dict
                Dictionary of data where the key is the class name and the values are the coordinates belongin to that
                class. This is fundamentally a classification problem!
        n_layers : int
                Number of hidden layers to use.
        units : int
                Number of units to use in each layer.
        epochs : int
                Number of epochs during training.
        activation : str
                Activation function to use.
        monitor : str
                Monitor parameter for use in training.
        lr : float
                Learning rate of the algorithm.
        batch_size : int
                batch size for the training.
        terminate_patience : int
                Patience value for termination to be used in a callback.
        lr_patience : int
                Patience in the learning rate reduction.
        """
        self.data_dict = data_dict
        self.n_layers = n_layers
        self.units = units
        self.epochs = epochs
        self.activation = activation
        self.monitor = monitor
        self.batch_size = batch_size
        self.lr = lr
        self.terminate_patience = terminate_patience
        self.lr_patience = lr_patience
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None
        self.model = None

    def _shuffle_and_split_data(self):
        """
        shuffle and split the parsed dataset

        Returns
        -------
        Updates the class state.
        """

        for key in self.data_dict:
            labels = tf.repeat(tf.convert_to_tensor(np.array(key, dtype=float)), len(self.data_dict[key]))
            stacked_data = tf.stack([self.data_dict[key], labels], axis=1)
            stacked_data = tf.random.shuffle(stacked_data)
            data_volume = len(stacked_data)
            train, test, validate = tf.split(stacked_data,
                                             [int(0.5 * data_volume), int(0.3 * data_volume), int(0.2 * data_volume)],
                                             axis=0)
            if self.train_ds is None:
                self.train_ds = train
            else:
                self.train_ds = tf.concat([self.train_ds, train], axis=0)
            if self.test_ds is None:
                self.test_ds = test
            else:
                self.test_ds = tf.concat([self.test_ds, test], axis=0)
            if self.val_ds is None:
                self.val_ds = validate
            else:
                self.val_ds = tf.concat([self.val_ds, validate], axis=0)

        self.train_ds = tf.random.shuffle(self.train_ds)
        self.train_ds = tf.random.shuffle(self.train_ds)
        self.train_ds = tf.random.shuffle(self.train_ds)

    def _build_model(self):
        """
        Build the ml model

        Returns
        -------
        Updates the class state.
        """

        model = tf.keras.Sequential()
        input_layer = Input(shape=[1])
        model.add(input_layer)

        # Loop over the layers excluding one for the input_layer, the second last, the embedding, and the softmax layer
        for i in range(self.n_layers - 2):
            model.add(tf.keras.layers.Dense(self.units, self.activation, kernel_regularizer=regularizers.l2(0.0000)))

        # Add the second-to-last layer
        model.add(tf.keras.layers.Dense(self.units,
                                        self.activation,
                                        name='second_last_layer',
                                        kernel_regularizer=regularizers.l2(0.0000)))
        # Add the embedding layer
        model.add(tf.keras.layers.Dense(self.units,
                                        self.activation,
                                        name='embedding_layer',
                                        kernel_regularizer=regularizers.l2(0.001)))

        # Add the softmax layer
        model.add(tf.keras.layers.Dense(len(self.data_dict), name='softmax_layer', activation='softmax'))

        self.model = model  # Add the model to the class

    def _compile_model(self) -> list:
        """
        Compile the model

        Returns
        -------
        callback_list : list
                A list of model callbacks to be applied during training.
        """

        # Define the callbacks
        terminating_callback = tf.keras.callbacks.EarlyStopping(monitor=self.monitor,
                                                                patience=self.terminate_patience, )
        reduction_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.monitor, patience=self.lr_patience,
                                                                  factor=0.1, mode='auto', min_lr=0.00001, cooldown=0)

        opt = tf.keras.optimizers.Adam(learning_rate=self.lr, decay=0.)

        self.model.compile(optimizer=opt, loss='categorical_crossentropy',
                           metrics=['acc'])

        return [terminating_callback, reduction_callback]

    def train_model(self) -> tf.data.Dataset:
        """
        Collect other methods and train the ML model

        Returns
        -------
        train_ds : tf.data.Dataset
                the training data set. Should be changed perhaps to the validation but this is not essential.
        """

        self._shuffle_and_split_data()  # Build the datasets
        self._build_model()  # Build the NN model
        callbacks = self._compile_model()
        print(self.model.summary())  # Print a model summary

        # Train the model
        for i in range(1, 6):
            self.model.fit(x=self.train_ds[:, 0],
                           y=tf.keras.utils.to_categorical(self.train_ds[:, 1]),
                           batch_size=self.batch_size,
                           shuffle=True,
                           validation_data=(self.test_ds[:, 0], tf.keras.utils.to_categorical(self.test_ds[:, 1])),
                           verbose=1,
                           epochs=self.epochs)

        return self.train_ds

    def _evaluate_model(self):
        """
        Evaluate the tensorflow model on the validation data

        Returns
        -------
        Prints the model evaluation.
        """

        attributes = self.model.evaluate(x=self.val_ds[:, 0],
                                         y=tf.keras.utils.to_categorical(self.val_ds[:, 1]))
        print(f"Loss: {attributes[0]} \n"
              f"Accuracy: {attributes[1]}")

    def get_embedding_layer_representation(self, data_array: np.ndarray) -> tf.Tensor:
        """
        Return the representation constructed by the embedding layer

        Parameters
        ---------
        data_array : np.array
                Data on which the model should be applied

        Returns
        -------
        predictions : tf.Tensor
                Predictions on the data_array returned in their high dimensional representation.
        """

        model = tf.keras.Sequential()
        input_data = Input(shape=[1])
        model.add(input_data)
        for layer in self.model.layers[:-1]:
            model.add(layer)

        model.build()
        return model.predict(data_array)
