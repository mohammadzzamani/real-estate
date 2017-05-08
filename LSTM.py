import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from DB_wrapper import DB_wrapper
from keras import optimizers
from keras import layers
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import random

import Util

# from sqlalchemy import create_engine
# from sqlalchemy.engine.url import URL

import DB_info

# Initially 8 columns
# columns from 2008/07 to 2016/09 = 6 + 84 + 9 = 99 columns
# We need 45 columns between 2011/11 and 2015/07
# So skip 8 initial columns + columns from 2008/07 to 2011/10 = 8 + 6 + 12*2 + 10 = 48 columns
INIT_SKIP = 48
MONTH_COLUMNS = 45
TRAIN_MONTHS = 43
# TEST_MONTHS = 9

# After all months, comes county column which is column number 99 + 8 = 107
#IP:
# COUNTY_COLUMN_NUMBER = 113
#MSP:
# COUNTY_COLUMN_NUMBER = 104
#SAF:
COUNTY_COLUMN_NUMBER = 107
LOOK_BACK = 30

# convert an array of values into a dataset matrix
def create_dataset(dataset, start , end, num_of_counties):


    dataX, dataY = [], []

    for j in xrange(num_of_counties):

        X = [ dataset[i:(i+LOOK_BACK), j] for i in xrange(start, end)]
        Y = [ dataset[ i + LOOK_BACK, j] for i  in xrange(start , end)]

        dataX.extend(X)
        dataY.extend(Y)

    return np.array(dataX), np.array(dataY)

# Get only county, month values
def get_county_month(dataset):
    county = dataset[:, COUNTY_COLUMN_NUMBER]
    county = np.reshape(county, (len(county), 1))
    month = dataset[:, INIT_SKIP: INIT_SKIP + MONTH_COLUMNS]

    dataset = np.concatenate((county, month), axis = 1)
    dataset = np.transpose(dataset)
    dataset = dataset.astype('float32')
    return dataset


def build_LSTM(trainX, trainY, testX, testY):
    print 'baseline: ', mean_squared_error(testY, testX[:, -1])
    batch_size = 25
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, LOOK_BACK, 1), return_sequences = True))
    model.add(layers.core.Dropout(0.2))
    model.add(LSTM(5,return_sequences=False))
    model.add(layers.core.Dropout(0.2))
    model.add(Dense(1))
    lr = 0.005
    decay = 0.95
    nb_epoch = 1
    adam = optimizers.adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)

    print "TrainX: ", trainX.shape
    print "TrainY: ", trainY.shape
    print "TestX: ", testX.shape
    print "TestY: ", testY.shape

    for i in range(nb_epoch):
        rd = random.random()
        if rd <0.95:
            adam.lr.set_value(lr)
        else:
            adam.lr.set_value(lr*2)

        print 'i: ' , i , ' lr: ' , adam.lr.get_value()
        model.fit(trainX, trainY, nb_epoch= 1, batch_size=batch_size, verbose=1, shuffle=True, validation_split= 0.15 )
        if i % 5 == 0:
            testPredict = model.predict(testX, batch_size=batch_size, verbose = 1)
            print 'lstm_i: ' , mean_squared_error(testY, testPredict)
        lr *= decay

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size, verbose = 1)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size, verbose = 1)


    yPrevTest = []

    for i in range(0, testY.shape[0] - 1):
         yPrevTest.append(testY[i])

    print len(yPrevTest)    

    testY = testY[1:]
    testPredict = testPredict[1:]
    
    testPredict = testPredict.reshape(testPredict.shape[0])
    yPrevTest = np.array(yPrevTest)

    print 'lstm - MSE: ' , mean_squared_error(testY, testPredict)
    print 'lstm - MAE: ' , mean_absolute_error(testY, testPredict)
    print ' test accuracy: ' , sum(1 for x,y in zip(np.sign(testPredict - yPrevTest),np.sign(testY - yPrevTest)) if x == y) / float(len(testY))

def get_train_and_test(dataset, train_size):

    num_of_months = dataset.shape[0]-1
    num_of_counties = dataset.shape[1]

    print 'num_of_months: ' , num_of_months , ' , num_of_counties: ' , num_of_counties
    print 'start: ' , 0 , ' , end: ', (train_size - LOOK_BACK-1)
    trainX, trainY = create_dataset(dataset,  start= 0 , end = train_size - LOOK_BACK-1, num_of_counties = num_of_counties )
    print 'start: ' , (train_size - LOOK_BACK-1), ' , end: ', ( num_of_months - LOOK_BACK-1)
    testX, testY = create_dataset(dataset,  start= train_size - LOOK_BACK-1 , end = num_of_months - LOOK_BACK-1, num_of_counties = num_of_counties)

    print 'data:'
    print trainX[1680,:]
    print trainX[1681,:]

    print "TrainX: ", trainX.shape
    print "TestX: ", testX.shape
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    trainX, trainY = Util.remove_nan(trainX, trainY)
    testX, testY = Util.remove_nan(testX, testY)

    return trainX, trainY, testX, testY


def build_lstm_on_labels():
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(DB_info.SAF_TABLE) #get_dataframe()
    dataset = Util.normalize_min_max(get_county_month(dataframe.values), TRAIN_MONTHS)

    print "Dataset shape: ", dataset.shape
    trainX, trainY, testX, testY = get_train_and_test(dataset, TRAIN_MONTHS)

    build_LSTM(trainX, trainY, testX, testY)


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)
    build_lstm_on_labels()
    print "--- Completed ---"