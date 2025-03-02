'''
This module implements the ActivationExplainer class, which generates
explanations for a Keras model from its training dataset using how the neurons
of the network activate for each train case (i.e. the activation patterns of
the training set).
'''
import keras
import numpy as np
import scipy as sp

class FastActivationExplainer:
    '''A simple explanation generator based on the activation pattern of the training cases of a Keras model. '''

    def __init__(self, model, x_train, verbose=0, n_layer=None):
        '''Initializes the explainer from the target model and its training dataset. It creates an activation pattern
        extractor and stores the activation patterns of the training cases.

        :param model: a Keras model.
        :type model: keras.Model
        :param x_train: the training data.
        :type x_train: numpy.array
        :param verbose: verbosity mode for the activation patterns extraction. 0=silent, 1=verbose.
        :type verbose: int
        '''
        self.model = model
        self.x_train = x_train
        self.verbose = verbose
        self.n_layer = n_layer
        inputs = self.model.inputs
        if n_layer is None:
            outputs = [layer.output for layer in self.model.layers]
        else:
            outputs = self.model.layers[n_layer].output

        self.transformer = keras.Model(inputs=inputs, outputs=outputs)

        # Activation pattern extraction
        self.x_train_act = self.transformer.predict(self.x_train, verbose=self.verbose)

    def get_distances(self, x, metric):
        '''Returns a numpy array with the distances between a new activation pattern and each training activation
        pattern. The distance between two activation patterns is the weighted sum of the distance (given by
        distance_function) between each layer:
            distance(activation1, activation2) = avg(distance_function(activation1.L, activation2.L) for L in layers)
        '''
        x_act = self.transformer.predict(x, verbose=self.verbose)
        if self.n_layer is None:
            return np.array([
                sp.spatial.distance.cdist(x_act_l.reshape(x_act_l.shape[0], -1), x_train_act_l.reshape(x_train_act_l.shape[0], -1), metric=metric)
                for (x_act_l, x_train_act_l) in zip(x_act, self.x_train_act)
            ])
        else:
            return sp.spatial.distance.cdist(x_act.reshape(x_act.shape[0], -1), self.x_train_act.reshape(self.x_train_act.shape[0], -1), metric=metric)

    def __get_average_distances(self, x, metric, distances=None, weights=None):
        '''Returns a numpy array with the distances between a new activation pattern and each training activation
        pattern. The distance between two activation patterns is the weighted sum of the distance (given by
        distance_function) between each layer:
            distance(activation1, activation2) = avg(distance_function(activation1.L, activation2.L) for L in layers)
        '''
        if distances is None:
            distances = self.get_distances(x, metric)
        if self.n_layer is None:
            distances = np.average(distances, weights=weights, axis=0)

        return np.array(distances)

    def explain(self, x, metric, distances=None, weights=None, top_k=None):
        if (np.array(x).ndim == 1):
            x = np.array([x])

        distances = self.__get_average_distances(x, metric, distances=distances, weights=weights)

        if top_k is not None and top_k < self.x_train.shape[0]:
            indices = np.argpartition(distances, top_k)[:, :top_k]
        else:
            indices = np.tile(np.arange(distances.shape[1]), (distances.shape[0], 1))
        distances = distances[np.arange(distances.shape[0])[:, None], indices]

        return indices, distances