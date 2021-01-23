""" Dense neural network model """

import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input



class DenseModel:
    """ Class for the construction and training of a dense NN model """

    def __init__(self, data_dict, n_layers=5, units=10, epochs=20, activation='relu', monitor='accuracy', lr=1e-4,
                 batch_size=100, terminate_patience=10, lr_patience=5):
        """ Python constructor """
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
        """ shuffle and split the parsed dataset """

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
        """ Build the ml model """

        model = tf.keras.Sequential()
        input = Input(shape=[1])
        model.add(input)

        # Loop over the layers excluding one for the input, the second last, the embedding, and the softmax layer
        for i in range(self.n_layers - 2):
            model.add(tf.keras.layers.Dense(self.units, self.activation, kernel_regularizer=regularizers.l2(0.0)))

        # Add the second-to-last layer
        model.add(tf.keras.layers.Dense(self.units,
                                        self.activation,
                                        name='second_last_layer',
                                        kernel_regularizer=regularizers.l2(0.0)))
        # Add the embedding layer
        model.add(tf.keras.layers.Dense(self.units,
                                        self.activation,
                                        name='embedding_layer',
                                        kernel_regularizer=regularizers.l2(0.0)))

        # Add the softmax layer
        model.add(tf.keras.layers.Dense(len(self.data_dict), name='softmax_layer', activation='softmax'))

        self.model = model  # Add the model to the class

    def _compile_model(self):
        """ Compile the model """

        # Define the callbacks
        terminating_callback = tf.keras.callbacks.EarlyStopping(monitor=self.monitor,
                                                                patience=self.terminate_patience, )
        reduction_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.monitor, patience=self.lr_patience,
                                                                  factor=0.1, mode='auto', min_lr=0.00001, cooldown=0)

        opt = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

        self.model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        return [terminating_callback, reduction_callback]

    def train_model(self):
        """ Collect other methods and train the ML model """

        self._shuffle_and_split_data()  # Build the datasets
        self._build_model()  # Build the NN model
        callbacks = self._compile_model()
        print(self.model.summary())  # Print a model summary

        # Train the model
        self.model.fit(x=self.train_ds[:, 0],
                       y=tf.keras.utils.to_categorical(self.train_ds[:, 1]),
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(self.test_ds[:, 0], tf.keras.utils.to_categorical(self.test_ds[:, 1])),
                       verbose=1,
                       epochs=self.epochs,
                       callbacks=callbacks)

        return self.train_ds

    def _evaluate_model(self):
        """ Evaluate the tensorflow model on the validation data """

        attributes = self.model.evaluate(x=self.val_ds[:, 0],
                                         y=tf.keras.utils.to_categorical(self.val_ds[:, 1]))
        print(f"Loss: {attributes[0]} \n"
              f"Accuracy: {attributes[1]}")
        # self.model.predict

    def get_embedding_layer_representation(self, data_array):
        """ Return the representation constructed by the embedding layer """


        model = tf.keras.Sequential()
        input = Input(shape=[1])
        model.add(input)
        for layer in self.model.layers[:-1]:
            model.add(layer)

        model.build()

        print(model.summary())

        return model.predict(data_array)