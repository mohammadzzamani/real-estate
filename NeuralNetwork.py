import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU, Merge
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNetwork:
    ID_SIZE = 2
    NUM_FEATURES = 50

    # Returns the ID columns as numpy ndarray
    def get_ids(self, dataframe):
        return dataframe.ix[:, 0: ID_SIZE].values


    # Combines all feature differences with features
    # and returns the combined features as numpy ndarray
    # Eg: Input: Feat1, Feat2, .. Featn
    # Output: Feat1, Feat2, ... Featn, Feat2-1, Feat3-2.. Featn-(n-1)
    def get_features(self, dataframe):
        # Extract the features as numpy ndarray
        features = dataframe.ix[:, ID_SIZE: NUM_FEATURES].values

        # Create ndarray for derived features (the differences)
        derived_features = np.diff(features)

        # Concatenate the actual features, and their differences
        X = np.concatenate((features, derived_features), axis = 0)

        return X


    # Returns the labels as a numpy ndarray
    def get_labels(self, dataframe):
        return dataframe[dateframe.columns[-1]].values


    # Standardize a given numpy ndarray (makes mean = 0)
    # x = (x - mean) / variance
    def standardize(self, values):
        # Mean is computed column wise
        mean = values.mean(axis = 0)
        variance = values.var(axis = 0)
        values = (values - mean) / variance


    # Normalizes the data (values reduced to 0 and 1)
    # x = (x - min) / (max - min)
    # Ideally: Use MixMaxScaler or any other normalizer provided by Pandas
    def normalize(self, values):
        minimum = values.min(axis = 0)
        maximum = values.max(axis = 0)
        values = -1 + 2 * (values - minimum) / (maximum - minimum)


    # Build the neural network
    def build_neural_network(self, xTrain, xTest, yTrain, yTest):
        model = Sequential()

        # Add layers
        model.add(Dense(output_dim = 50, input_dim = len(xTrain[0]), init = 'normal', activation = 'sigmoid'))
        model.add(Dense(output_dim = 10, init = 'normal', activation = 'tanh'))
        model.add(Dense(output_dim = 1, init = 'normal'))

        # Compile model
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')

        model.fit(xTrain, yTrain, nb_epoch = 10, batch_size = 100, validation_split = 0.1)

        score = model.evaluate(xTest, yTest, batch_size = 100)
        prediction = model.predict(xTest, batch_size = 100, verbose = 1)

        result = [(yTest[i], prediction[i][0] for i in xrange(0, 30))]