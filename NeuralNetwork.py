import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU, Merge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from DB_wrapper import  DB_wrapper

DATABASE = 'mztwitter'
TRAIN_TABLE_NAME = 'NLP_train_features'
TEST_TABLE_NAME = 'NLP_test_features'

class NeuralNetwork:
    ID_SIZE = 1

    # Change this depending on whatever is the number of features
    # in the dataframe.
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
        X = np.concatenate((features, derived_features), axis = 1)

        return X


    # Returns the labels as a numpy ndarray
    def get_labels(self, dataframe):
        return dataframe[dataframe.columns[-1]].values


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

        result = [(yTest[i], prediction[i][0]) for i in xrange(0, 30)]

    def __init__(self):
        print "-- Created NeuralNetwork Object --"


# def get_dataframe(db, table):
#     # Create SQL engine
#     myDB = URL(drivername='mysql', database=db, query={
#             'read_default_file' : '/home/pratheek/.my.cnf' })
#     engine = create_engine(name_or_url=myDB)
#     engine1 = engine
#     connection = engine.connect()
#
#     query = connection.execute('select * from %s' % table)
#     df_feat = pd.DataFrame(query.fetchall())
#     df_feat.columns = query.keys()
#
#     return df_feat

#def helper():
if __name__ == "__main__":
    # get dataframe for train and test
    # get xTrain and xTest from these dataframes (get_features())
    # get yTrain and yTest from these dataframes (get_labels())
    # build neural network (build_neural_network())
    db_wrapper = DB_wrapper()
    dataframe_train = db_wrapper.retrieve_data(TRAIN_TABLE_NAME) #get_dataframe(DATABASE, TRAIN_TABLE_NAME)
    #dataframe_test = db_wrapper.retrieve_data(TEST_TABLE_NAME) #get_dataframe(DATABASE, TEST_TABLE_NAME)

    # generating random labels for now <-- This is not required though
    # labels will be part of table, so use the main dataframe
    label_frame = pd.DataFrame(np.random.uniform(-1, 1, size = (len(dataframe_train), 1)))
    dataframe_train = pd.concat([dataframe_train, label_frame], axis = 1)

    train_size = 10000

    train_set = dataframe_train.ix[0: train_size, :]
    test_set = dataframe_train.txt[train_size:, :]

    Network = NeuralNetwork()
    xTrain = Network.get_features(train_set)
    yTrain = Network.get_labels(train_set)
    xTest = Network.get_features(test_set)
    yTest = Network.get_labels(test_set)

    Network.build_neural_network(xTrain, xTest, yTrain, yTest)
    print "--- Completed ---"