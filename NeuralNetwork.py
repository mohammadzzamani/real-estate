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
from sklearn.svm import SVR
from sklearn import linear_model

# DATABASE = 'mztwitter'
# TRAIN_TABLE_NAME = 'NLP_features_msp'
# TEST_TABLE_NAME = 'NLP_test_features_saf'
ID_SIZE = 0

# Change this depending on whatever is the number of features
# in the dataframe.
NUM_FEATURES = 34
TOTAL_MONTHS = 45

MONTH = 'month'

class NeuralNetwork_:

    # Returns the ID columns as numpy ndarray
    def get_ids(self, dataframe):
        return dataframe.ix[:, 0: ID_SIZE].values


    def add_diff_features(self, df ,train_month ):
        columns = df.columns

        cols = np.append(columns[:-1],  'prev_month')
        cols = np.append(cols , columns[-1])
        print 'new cols: ' , cols
        # print df.shape
        test_data = np.empty((0, df.shape[1]+1))
        train_data = np.empty((0, df.shape[1]+1))
        print 'train_data.shape: ' , train_data.shape
        tr_indices = []
        te_indices = []
        for index , row in df.iterrows():
            if row.month <> 0:
                prev_index = str(int(row.cnty))+'_'+str(int(row.month)-1)
                prev_data = df.ix[prev_index].values
                current_data = row.values
                diff_data = current_data[:NUM_FEATURES] -  prev_data[:NUM_FEATURES]
                other_data = current_data[NUM_FEATURES:]
                diff_data = np.append(diff_data,other_data)
                diff_data = np.append(diff_data, prev_data[len(prev_data)-1])
                if row.month > train_month:
                    test_data = np.vstack((test_data , diff_data))
                    te_indices.append(index)
                else:
                    train_data = np.vstack((train_data , diff_data))
                    tr_indices.append(index)

        train_df = pd.DataFrame(data = train_data	, index = tr_indices, columns = cols)
        test_df = pd.DataFrame(data = test_data       , index = te_indices, columns = cols)

        train_df.reset_index(drop = True, inplace = True)
        test_df.reset_index(drop = True, inplace = True)
        return [train_df , test_df]


    def add_prev_features(self, df ,train_month ):
        columns = df.columns

        cols = np.append(columns[:-1],  'prev_month')
        cols = np.append(cols , columns[-1])
        print 'new cols: ' , cols
        # print df.shape
        test_data = np.empty((0, df.shape[1]+1))
        train_data = np.empty((0, df.shape[1]+1))
        print 'train_data.shape: ' , train_data.shape
        tr_indices = []
        te_indices = []
        for index , row in df.iterrows():
            [cnty , month ]   = index.split('_')
            
            if month <> 0:
                prev_index = str(int(cnty))+'_'+str(int(month)-1)
                prev_data = df.ix[prev_index].values
                current_data = row.values
                diff_data = current_data[:NUM_FEATURES] -  prev_data[:NUM_FEATURES]
                other_data = current_data[NUM_FEATURES:]
                diff_data = np.append(diff_data,other_data)
                diff_data = np.append(diff_data, prev_data[len(prev_data)-1])
                if month > train_month:
                    test_data = np.vstack((test_data , diff_data))
                    te_indices.append(index)
                else:
                    train_data = np.vstack((train_data , diff_data))
                    tr_indices.append(index)

        train_df = pd.DataFrame(data = train_data	, index = tr_indices, columns = cols)
        test_df = pd.DataFrame(data = test_data       , index = te_indices, columns = cols)

        train_df.reset_index(drop = True, inplace = True)
        test_df.reset_index(drop = True, inplace = True)
        return [train_df , test_df]



    # Combines all feature differences with features
    # and returns the combined features as numpy ndarray
    # Eg: Input: Feat1, Feat2, .. Featn
    # Output: Feat1, Feat2, ... Featn, Feat2-1, Feat3-2.. Featn-(n-1)
    def get_features(self, dataframe):
        # Extract the features as numpy ndarray
        features = dataframe.ix[:, ID_SIZE: ID_SIZE + NUM_FEATURES]
        print "Rows features: ", features.shape
        return features


    # Returns the labels as a numpy ndarray
    def get_labels(self, dataframe):
        return dataframe[dataframe.columns[-1]].values # - dataframe[dataframe.columns[-1]].values


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

    def baseline_model(self,xTrain, xTest, yTrain, yTest):
        # create model
        model = Sequential()
        model.add(Dense(100, input_dim=len(xTrain[0]) , init='normal', activation='tanh'))
        model.add(Dense(output_dim = 20, init='normal' , activation = 'relu'))
        model.add(Dense(output_dim = 5, init='normal' , activation = 'linear'))
        model.add(Dense(1, init='normal'))
        # Compile model

        lr = 0.1
        decay = 0.975
        adam = optimizers.adam(lr = lr, decay = decay)
        model.compile(loss = 'mean_absolute_error', optimizer = adam)
        nb_epochs = 500
        for i in xrange(nb_epochs):
            lr = lr * decay
            adam.lr.set_value(lr)
            model.fit(xTrain, yTrain, nb_epoch = 1, batch_size = 5000, shuffle = True, validation_split = 0.1, verbose = 1)

            if i %10 == 0:
                score = model.evaluate(xTest, yTest, batch_size = 5000)
                print 'score: ' , score

                testPredict = model.predict(xTest, batch_size = 5000, verbose = 1)
                    print 'Neural Network_i: ', mean_squared_error(yTest, testPredict)
                    print 'Neural Network_i: ', mean_absolute_error(yTest, testPredict)
                print ' test accuracy: ' , sum(1 for x,y in zip(np.sign(testPredict),np.sign(yTest)) if x == y) / float(len(yTest))


    # Build the neural network
    def build_neural_network(self, xTrain, xTest, yTrain, yTest):

        print "---- In Build Neural Network ----"
        print "xTrain: ", xTrain.shape
        print "yTrain: ", yTrain.shape
        print "xTest: ", xTest.shape
        print "yTest: ", yTest.shape

        model = Sequential()

        # Add layers
        model.add(Dense(output_dim = 15, input_dim = len(xTrain[0]), init='normal', activation = 'tanh'))
        # model.add(layers.core.Dropout(0.2))
        model.add(Dense(output_dim = 5, init='normal' , activation = 'relu'))
        #model.add(layers.core.Dropout(0.2))
        # model.add(BatchNormalization())
        model.add(Dense(output_dim = 1, init='normal' , activation = 'linear'))
        #model.add(layers.core.Dropout(0.2))
        # model.add(BatchNormalization())
        #model.add(Dense(output_dim = 1, init='normal', activation = 'sigmoid'))
        # label_model = Sequential()
        # label_model.add(Dense(output_dim = 1, input_dim = 1, init= 'normal', activation = 'sigmoid'))
        #
        # final_model = Sequential()
        # final_model.add(Merge([model, label_model], mode = 'concat'))
        # final_model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

        lr = 1.0

        adam = optimizers.adam(lr = lr)
        model.compile(loss = 'mean_squared_error', optimizer = adam)
        #model.compile(loss='categorical_crossentropy', optimizer=adam , metrics=['accuracy'])
        nb_epochs = 200
        decay = 0.95
        for i in xrange(nb_epochs):

            # Compile model
            rd = random.random()
            if rd < 0.9:
                adam.lr.set_value(lr)
            else:
                 adam.lr.set_value(lr * 2)
            print 'i: ' , i , ' lr:  ' , adam.lr.get_value()

            model.fit(xTrain, yTrain, nb_epoch = 1, batch_size = 5000, shuffle = True, validation_split = 0.1, verbose = 1)

        # final_model.fit([xTrain[:, :-1], xTrain[:,-1]], yTrain, nb_epoch = 1, batch_size = 100, shuffle = True, validation_split = 0.15)
            if i % 10 == 0:
                testPredict = model.predict(xTest, batch_size = 5000, verbose = 1)
                # testPredict = final_model.predict([xTest[:,:-1], xTest[:, -1]], batch_size = 100, verbose = 1)
                print 'Neural Network_i: ', mean_squared_error(yTest, testPredict)
        print 'Neural Network_i: ', mean_absolute_error(yTest, testPredict)

        print ' test accuracy: ' , sum(1 for x,y in zip(testPredict,yTest) if x == y) / float(len(yTest))
        lr = lr * decay


        score = model.evaluate(xTest, yTest, batch_size = 100)
        prediction = model.predict(xTest, batch_size = 100, verbose = 1)
        # prediction = final_model.predict([xTest[:,:-1], xTest[:, -1]], batch_size = 100, verbose = 1)
        print 'Result: ', mean_squared_error(yTest, prediction)

        result = [(yTest[i], prediction[i][0]) for i in xrange(0, 30)]



    def compute_baseline(self, mean ,  test_set):
        print 'compute_baseline:'
        p_month = []
        c_month = []
        for index, row in test_set.iterrows():
            previous_month = row.prev_month
            current_month = row.label
            if previous_month is  None or current_month is  None or  math.isnan(previous_month) or math.isnan(current_month):
                continue
            p_month.append(previous_month)
            c_month.append(current_month)

        dif = [c_month[i] - p_month[i] for i in xrange(len(c_month))]
        m_month = [mean+i for i in p_month]

        print 'baseline1 (MAE): ' , mean_absolute_error(p_month, c_month)
        print 'baseline1 (MSE): ', mean_squared_error(p_month, c_month)
        print 'baseline2 (MAE):  ', mean_absolute_error(m_month, c_month)
        print 'baseline2 (MSE): ', mean_squared_error(m_month, c_month)
        print ' test accuracy: ' , sum(1 for x,y in zip(np.sign([mean for i in c_month]),np.sign(dif)) if x == y) / float(len(c_month))

    def __init__(self):
        print "-- Created NeuralNetwork Object --"



    def linear_model(self, xTrain, yTrain, xTest, yTest):
        lr = linear_model.LinearRegression()
        lr.fit(xTrain, yTrain)
        lr_pred_test = lr.predict(xTest)
        lr_pred_train = lr.predict(xTrain)

        print 'Result_test: ', mean_squared_error(yTest, lr_pred_test)
        print 'Result_train: ', mean_squared_error(yTrain, lr_pred_train)

        print 'Result_test: ', mean_absolute_error(yTest, lr_pred_test)
        print 'Result_train: ', mean_absolute_error(yTrain, lr_pred_train)

        lr_pred_test = np.sign(lr_pred_test)
        lr_pred_train = np.sign(lr_pred_train )

        yTest_c =  np.sign(yTest)
        yTrain_c = np.sign(yTrain)

        print ' lr test accuracy: ' , sum(1 for x,y in zip(lr_pred_test,yTest) if x == y) / float(len(yTest))
        print ' lr train accuracy: ' , sum(1 for x,y in zip(lr_pred_train,yTrain) if x == y) / float(len(yTrain))

        print  'lr.coef_: '
        print lr.coef_



if __name__ == "__main__":
    # get dataframe for train and test
    # get xTrain and xTest from these dataframes (get_features())
    # get yTrain and yTest from these dataframes (get_labels())
    # build neural network (build_neural_network())
    db_wrapper = DB_wrapper()
    dataframe_train = db_wrapper.retrieve_data(DB_info.FEATURE_TABLE) #get_dataframe(DATABASE, TRAIN_TABLE_NAME)
    dataframe_train = dataframe_train.set_index('cnty_month')

    dataframe_train = Util.normalize_each_county(dataframe_train, TOTAL_MONTHS,  NUM_FEATURES)


    Network = NeuralNetwork_()
    print 'dataframe_train before adding diff: ' , dataframe_train.shape
    [train_set , test_set] = Network.add_diff_features(dataframe_train, 0.8 * TOTAL_MONTHS )



    mean = np.mean(train_set.label) - np.mean(train_set.prev_month)
    Network.compute_baseline(mean, test_set)


    xTrain = train_set.ix[:, ID_SIZE: ID_SIZE + NUM_FEATURES].values
    #xTrain = train_set.ix[:, ID_SIZE: ID_SIZE + NUM_FEATURES+1].values
    yTrain = train_set[train_set.columns[-1]].values
    xTest = test_set.ix[:, ID_SIZE: ID_SIZE + NUM_FEATURES].values
    #xTest = test_set.ix[:, ID_SIZE: ID_SIZE + NUM_FEATURES+1].values
    yTest = test_set[test_set.columns[-1]].values

    xTrain, yTrain = Util.remove_nan(xTrain, yTrain)
    xTest, yTest = Util.remove_nan(xTest, yTest)


    yTest = yTest - xTest[:,-1]
    yTrain = yTrain - xTrain[:,-1]

    xTrain = xTrain [: , :-1]
    xTest = xTest [: , : -1]




    #linear regression
    Network.linear_model( xTrain, yTrain, xTest, yTest)


    '''
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_lin.fit(xTrain, yTrain)
    svr_lin_test = svr_lin.predict(xTest)
    svr_lin_train = svr_lin.predict(xTrain)
    print 'svr_lin_test: ', mean_absolute_error(yTest, svr_lin_test)
    print 'svr_lin_train: ', mean_absolute_error(yTrain, svr_lin_train)




    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_poly.fit(xTrain, yTrain)
    svr_poly_test = svr_poly.predict(xTest)
    svr_poly_train = svr_poly.predict(xTrain)
    print 'svr_poly_test: ', mean_absolute_error(yTest, svr_poly_test)
    print 'svr_poly_train: ', mean_absolute_error(yTrain, svr_poly_train)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr = svr_rbf.fit(xTrain, yTrain)
    y_rbf_test = svr.predict(xTest)
    y_rbf_train = svr.predict(xTrain)
    #print 'coeff: ' , svr.coef_
    print 'Result_test: ', mean_squared_error(yTest, y_rbf_test)
    print 'Result_train: ', mean_squared_error(yTrain, y_rbf_train)

    print 'Result_test: ', mean_absolute_error(yTest, y_rbf_test)
    print 'Result_train: ', mean_absolute_error(yTrain, y_rbf_train)
    '''

    Network.baseline_model( xTrain, xTest, yTrain, yTest)
    #Network.build_neural_network(xTrain, xTest, yTrain, yTest)
    print "--- Completed ---"

