import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from DB_wrapper import DB_wrapper

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

import DB_info

# Initially 8 columns
# columns from 2008/07 to 2016/09 = 6 + 84 + 9 = 99 columns
# We need 45 columns between 2011/11 and 2015/07
# So skip 8 initial columns + columns from 2008/07 to 2011/10 = 8 + 6 + 12*2 + 10 = 48 columns
INIT_SKIP = 48
MONTH_COLUMNS = 45
TRAIN_MONTHS = 36
TEST_MONTHS = 9

# After all months, comes county column which is column number 99 + 8 = 107
COUNTY_COLUMN_NUMBER = 107
LOOK_BACK = 12

# convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#
#     return np.array(dataX), np.array(dataY)
def create_dataset(dataset, train_test_split):

    trainX, trainY, testX , testY = [], [], [], []

    for j in xrange(len(dataset)):
        X = [ dataset[j,i:(i+LOOK_BACK)] for i in xrange(len(dataset[j])-LOOK_BACK-1)]
        Y = [ dataset[j, i + LOOK_BACK] for i  in xrange(len(dataset[j])-LOOK_BACK-1) ]

        if j < train_test_split:
            trainX.extend(X)
            trainY.extend(Y)
        else:
            testX.extend(X)
            testY.extend(Y)

    return np.array(trainX), np.array(trainY), np.array(testX) , np.array(testY)


# Get only county, month values
def get_county_month(dataset):
    county = dataset[:, COUNTY_COLUMN_NUMBER]
    county = np.reshape(county, (len(county), 1))
    month = dataset[:, INIT_SKIP: INIT_SKIP + MONTH_COLUMNS]

    dataset = np.concatenate((county, month), axis = 1)
    # dataset = np.transpose(dataset)
    dataset = dataset.astype('float32')
    return dataset


# This method normalizes the data using the mean and
# standard deviation, obtained using the train data
# only.
# Note: This happens column wise, because the data has
# been transposed.
# Data is in the form:
# County ......
# Month1 ......
# MonthN ......
def normalize(dataset, train_size):
    mean = np.mean(dataset[1: train_size], axis = 0)
    standard_deviation = np.std(dataset[1: train_size], axis = 0)
    dataset = (dataset - mean)/standard_deviation
    return dataset


def build_LSTM(trainX, trainY, testX, testY, look_back):
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    print "TrainX: ", trainX.shape
    print "TrainY: ", trainY.shape
    print "TestX: ", testX.shape
    print "TestY: ", testY.shape

    for i in range(100):
        model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)


def get_train_and_test(dataset, train_size, num_of_months):

    # reshape into X=t and Y=t+1
    trainX, trainY, testX , testY = create_dataset(dataset,  train_test_split= train_size)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
  
    return trainX, trainY, testX, testY


def build_lstm_on_labels():
    db_wrapper = DB_wrapper()                       
    dataframe = db_wrapper.retrieve_data(DB_info.SAF_TABLE) #get_dataframe()
    dataset = normalize(get_county_month(dataframe.values), TRAIN_MONTHS)

    print "Dataset shape: ", dataset.shape
                             
    # split into train and test sets
    train_size = TRAIN_MONTHS
    num_of_months = dataset.shape[1]-1
    trainX, trainY, testX, testY = get_train_and_test(dataset, train_size, num_of_months)
    build_LSTM(trainX, trainY, testX, testY, LOOK_BACK)


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)
    build_lstm_on_labels()
    print "--- Completed ---"
