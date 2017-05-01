import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU, Merge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import layers, optimizers
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from DB_wrapper import  DB_wrapper
from keras.layers.normalization import BatchNormalization
import math
import random
import Util
import DB_info

# DATABASE = 'mztwitter'
# TRAIN_TABLE_NAME = 'NLP_features_msp'
# TEST_TABLE_NAME = 'NLP_test_features_saf'
ID_SIZE = 0

# Change this depending on whatever is the number of features
# in the dataframe.
NUM_FEATURES = 40
TOTAL_MONTHS = 45

MONTH = 'month'

class NeuralNetwork:
    
    # Returns the ID columns as numpy ndarray
    def get_ids(self, dataframe):
        return dataframe.ix[:, 0: ID_SIZE].values


    def add_derived_features(self, dataframe):
        features = dataframe.ix[:, : NUM_FEATURES]
        other_info = dataframe.ix[:, NUM_FEATURES:]

        print "************* Other_info: ", other_info.shape

        features = features.values
        derived_features = np.empty((0, features.shape[1]), dtype = float)
        print "derived shape: ", derived_features.shape
        print derived_features
        
        # Generate the derived features
        for i in range(0, len(features)):
            if int(other_info.ix[i, MONTH]) == 0:
                # Skip the month zero
                derived_features = np.append(derived_features, np.zeros((1, NUM_FEATURES)), axis = 0)
            else:
                difference = np.diff((features[i - 1], features[i]), axis = 0)
                derived_features = np.append(derived_features, difference, axis = 0)
        
        #features = np.append(features, derived_features, axis = 1)
        #features = np.append(features, other_info.values, axis = 1)
        #dataframe = pd.DataFrame(features)
        
        features = pd.DataFrame(np.append(features, derived_features, axis = 1))
        features.reset_index(drop = True, inplace = True)
        other_info.reset_index(drop = True, inplace = True)
        dataframe = pd.concat([features, other_info], axis = 1)
        dataframe = dataframe[dataframe.month != 0]
        print "Dataframe shape: ", dataframe.shape
        return dataframe


    # Combines all feature differences with features
    # and returns the combined features as numpy ndarray
    # Eg: Input: Feat1, Feat2, .. Featn
    # Output: Feat1, Feat2, ... Featn, Feat2-1, Feat3-2.. Featn-(n-1)
    def get_features(self, dataframe):
        # Extract the features as numpy ndarray
        features = dataframe.ix[:, ID_SIZE: ID_SIZE + 2 * NUM_FEATURES]
        prev_month = dataframe.prev_month.values
        prev_month = np.reshape(prev_month, (prev_month.shape[0], 1))

        print "Rows features: ", features.shape
        print "Prev Month: ", prev_month.shape

        # Create np array for derived features (the differences)
        # where each row represents the difference between the
        # features of a county's month(i+1) and the same county's
        # month(i). This is not computed for month 0.
        #diff_features = np.empty((0, features.shape[1]))
        #for i in range(0, len(features) - 1):
        #    if (i % TOTAL_MONTHS == 0):
        #        diff_features = np.append(diff_features, np.zeros((1, NUM_FEATURES)))
        #    else:
        #        diff_features = np.append(diff_features, np.diff((features[i], features[i +1]), axis = 0), axis = 0)


        # Concatenate the actual features, and their differences
        # X = np.concatenate((features, derived_features), axis = 1)
        # X = np.concatenate((X, prev_month), axis = 1)
        X = np.concatenate((features, prev_month), axis = 1)

        return X


    # Returns the labels as a numpy ndarray
    def get_labels(self, dataframe):
        return dataframe[dataframe.columns[-2]].values # - dataframe[dataframe.columns[-1]].values


    # Standardize a given numpy ndarray (makes mean = 0)
    # x = (x - mean) / variance
    def standardize(self, values):
        # Mean is computed column wise
        mean = values.mean(axis = 0)
        variance = values.var(axis = 0)
        values = (values - mean) / variance
        return values


    # Normalizes the data (values reduced to 0 and 1)
    # x = (x - min) / (max - min)
    # Ideally: Use MixMaxScaler or any other normalizer provided by Pandas
    def normalize(self, values):
        minimum = values.min(axis = 0)
        maximum = values.max(axis = 0)
        values = -1 + 2 * (values - minimum) / (maximum - minimum)
        return values


    # Build the neural network
    def build_neural_network(self, xTrain, xTest, yTrain, yTest):
        print "---- In Build Neural Network ----"
        print "xTrain: ", xTrain.shape
        print "yTrain: ", yTrain.shape
        print "xTest: ", xTest.shape
        print "yTest: ", yTest.shape

        model = Sequential()

        # Add layers
        model.add(Dense(output_dim = 150, input_dim = len(xTrain[0]), init='normal', activation = 'linear'))
        model.add(layers.core.Dropout(0.2))
        model.add(Dense(output_dim = 60, init='normal' , activation = 'relu'))
        model.add(layers.core.Dropout(0.2))
        # model.add(BatchNormalization())
        model.add(Dense(output_dim = 20, init='normal' , activation = 'linear'))
        model.add(layers.core.Dropout(0.2))
        # model.add(BatchNormalization())
        model.add(Dense(output_dim = 1, init='normal'))

        # label_model = Sequential()
        # label_model.add(Dense(output_dim = 1, input_dim = 1, init= 'normal', activation = 'sigmoid'))
        #
        # final_model = Sequential()
        # final_model.add(Merge([model, label_model], mode = 'concat'))
        # final_model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

        lr = 0.1

        adam = optimizers.adam(lr = lr)
        model.compile(loss = 'mean_squared_error', optimizer = adam)

        nb_epochs = 10000
        decay = 0.98
        for i in xrange(nb_epochs):



            # Compile model
            # rd = random.random()
            # if rd < 0.95:
            adam.lr.set_value(lr)
            # else:
            #     adam.lr.set_value(lr * 2)
            print 'i: ' , i , ' lr:  ' , adam.lr.get_value()

            model.fit(xTrain, yTrain, nb_epoch = 1, batch_size = 100, shuffle = True, validation_split = 0.1)
            # final_model.fit([xTrain[:, :-1], xTrain[:,-1]], yTrain, nb_epoch = 1, batch_size = 100, shuffle = True, validation_split = 0.15)
            if i % 10 == 0:
                testPredict = model.predict(xTest, batch_size = 100, verbose = 1)
                # testPredict = final_model.predict([xTest[:,:-1], xTest[:, -1]], batch_size = 100, verbose = 1)
                print 'Neural Network_i: ', mean_squared_error(yTest, testPredict)


            if i% 25 == 0:
                lr = lr * decay

        
        score = model.evaluate(xTest, yTest, batch_size = 100)
        prediction = model.predict(xTest, batch_size = 100, verbose = 1)
        # prediction = final_model.predict([xTest[:,:-1], xTest[:, -1]], batch_size = 100, verbose = 1)
        print 'Result: ', mean_squared_error(yTest, prediction)

        result = [(yTest[i], prediction[i][0]) for i in xrange(0, 30)]


    def add_prev_month_value(self, dataframe):
        dataframe['prev_month'] = None
        #dataframe = dataframe_train.set_index('cnty_month')

        for index, row in dataframe.iterrows():
            splitted = index.split('_')
            cnty = splitted[0]
            month = int(splitted[1]) -1
            # print cnty , ' , ', month
            if month >= 0:
                prev_month_id =  cnty+'_'+str(month)
                prev_month = dataframe.label[prev_month_id]
                dataframe.set_value(index, 'prev_month', prev_month)

        print list(dataframe.columns.values)
        return dataframe


    def compute_baseline(self, test_set):
        previous_month = test_set.prev_month.values
        current_month = test_set.label.values
        p_month = []
        c_month = []

        for i in xrange(len(previous_month)):
            if previous_month[i] is  None or current_month[i] is  None or  math.isnan(previous_month[i]) or math.isnan(current_month[i]):
                continue
            p_month.append(previous_month[i])
            c_month.append(current_month[i])

        print 'baseline: ', mean_squared_error(p_month, c_month)


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
    dataframe_train = db_wrapper.retrieve_data(DB_info.FEATURE_TABLE) #get_dataframe(DATABASE, TRAIN_TABLE_NAME)
    dataframe_train = dataframe_train.set_index('cnty_month')
    dataframe_train = dataframe_train.drop(dataframe_train.columns[[i for i in xrange(NUM_FEATURES/2, NUM_FEATURES)]], axis=1)
    old_names = dataframe_train.columns.values()
    new_names = ['feat_'+str(i) for i in xrange(NUM_FEATURES)]
    dataframe_train.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    print dataframe_train.columns
    dataframe_train = Util.normalize_each_county(dataframe_train, TOTAL_MONTHS,  NUM_FEATURES)

    # dataframe_train = dataframe_train.set_index('cnty_month')
    # for index, row in dataframe_train.iterrows():
    #     splitted = index.split('_')
    #     cnty = splitted[0]
    #     month = int(splitted[1]) -1
    #     # print cnty , ' , ', month
    #     if month >= 0:
    #         prev_month_id =  cnty+'_'+str(month)
    #         prev_month = dataframe_train.label[prev_month_id]
    #         dataframe_train.set_value(index, 'prev_month', prev_month)
    #         # print dataframe_train.label[prev_month_id], ' , ' , dataframe_train.prev_month[index] , ' , ', dataframe_train.label[index]
    # # dataframe_train = dataframe_train.drop('prev_month')
    # print list(dataframe_train.columns.values)




    #dataframe_test = db_wrapper.retrieve_data(TEST_TABLE_NAME) #get_dataframe(DATABASE, TEST_TABLE_NAME)

    # generating random labels for now <-- This is not required though
    # labels will be part of table, so use the main dataframe
    #label_frame = pd.DataFrame(np.random.uniform(-1, 1, size = (len(dataframe_train), 1)))
    #dataframe_train = pd.concat([dataframe_train, label_frame], axis = 1)
    Network = NeuralNetwork()
    dataframe_train = Network.add_prev_month_value(dataframe_train)
    dataframe_train = Network.add_derived_features(dataframe_train)


    print "Total rows in Dataset: ", len(dataframe_train)
    print "Total Columns in Dataset: ", len(dataframe_train.columns)
    train_size = int(0.8 * len(dataframe_train))

    train_set = dataframe_train.ix[0: train_size, :]
    test_set = dataframe_train.ix[train_size:, :]


    # previous_month = test_set.prev_month.values
    # current_month = test_set.label.values
    # p_month = []
    # c_month = []
    # for i in xrange(len(previous_month)):
    #     if previous_month[i] is  None or current_month[i] is  None or  math.isnan(previous_month[i]) or math.isnan(current_month[i]):
    #         continue
    #     p_month.append(previous_month[i])
    #     c_month.append(current_month[i])

    # print 'baseline: ', mean_squared_error(p_month, c_month)
    Network.compute_baseline(test_set)

    # print 'length: ' , len(test_set.prev_month.values), ' , ', len(test_set.label.values)
    # print 'length: ' , len(p_month), ' , ', len(c_month)

    xTrain = Network.get_features(train_set)
    yTrain = Network.get_labels(train_set)
    xTest = Network.get_features(test_set)
    yTest = Network.get_labels(test_set)

    xTrain, yTrain = Util.remove_nan(xTrain, yTrain)
    xTest, yTest = Util.remove_nan(xTest, yTest)


    yTest = yTest - xTest[:,-1]
    yTrain = yTrain - xTrain[:,-1]

    xTrain = xTrain [: , :-1]
    xTest = xTest [: , : -1]



    xTrain = xTrain[:2000, :]
    # xTest = xTest[0:1000, :]
    yTrain = yTrain[:2000]
    # yTest = yTest[0:1000]


    Network.build_neural_network(xTrain, xTest, yTrain, yTest)
    print "--- Completed ---"
