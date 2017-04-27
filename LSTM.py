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
from keras import optimizers
from keras import layers
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        X = [ dataset[i:(i+LOOK_BACK), j] for i in xrange(start, end)] # len(dataset[j])-LOOK_BACK-1)]
        Y = [ dataset[ i + LOOK_BACK, j] for i  in xrange(start , end)]

        # X = [ dataset[j,i:(i+LOOK_BACK)] for i in xrange(len(dataset[j])-LOOK_BACK-1)]
        # Y = [ dataset[j, i + LOOK_BACK] for i  in xrange(len(dataset[j])-LOOK_BACK-1) ]
        dataX.extend(X)
        dataY.extend(Y)
        # else:
        #     testX.extend(X)
        #     testY.extend(Y)

    return np.array(dataX), np.array(dataY) #, np.array(testX) , np.array(testY)


    # for i in range(len(dataset)-look_back-1):
    #     a = dataset[i:(i+look_back), 0]
    #     dataX.append(a)
    #     dataY.append(dataset[i + look_back, 0])
    #
    # return np.array(dataX), np.array(dataY)


# Get only county, month values
def get_county_month(dataset):
    county = dataset[:, COUNTY_COLUMN_NUMBER]
    county = np.reshape(county, (len(county), 1))
    month = dataset[:, INIT_SKIP: INIT_SKIP + MONTH_COLUMNS]

    dataset = np.concatenate((county, month), axis = 1)
    dataset = np.transpose(dataset)
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
    print 'mean: ' , mean.shape
    standard_deviation = np.std(dataset[1: train_size], axis = 0)
    print 'standard_deviation: ' , standard_deviation.shape
    print 'dataset: ' , dataset.shape
    dataset = (dataset - mean) * 10.0/standard_deviation
    return dataset

# def normalize(dataset, train_size):
#     minimum = np.min(dataset[1: train_size], axis = 0)
#     maximum = np.max(dataset[1: train_size], axis = 0)
#     print 'minimum'
#     # print minimum[1:20]
#     print 'maximum'
#     # print maximum[1:20]
#     print 'min: ' , minimum.shape
#     # standard_deviation = np.std(dataset[1: train_size], axis = 0)
#     # print 'standard_deviation: ' , standard_deviation.shape
#     # print 'dataset: ' , dataset.shape
#     dataset = ((dataset - minimum)* 100.0)/ (maximum-minimum)
#     # dataset = (dataset - mean)/standard_deviation
#     return dataset


def do_pca(trainX, trainY, testX, testY):
    pca = PCA(n_components=2)

    trainX = trainX.reshape(trainX.shape[0] , trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0])
    pca.fit(trainX)
    trainX_pca = pca.fit_transform(trainX)
    # testX_pca = pca.fit_transform(testX)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trainX_pca[:,0], trainX_pca[:,1], -trainY, zdir='z', c= 'red')
    plt.savefig("demo.png")


def build_LSTM(trainX, trainY, testX, testY):
    print 'baseline: ', mean_squared_error(testY, testX[:, -1])
    batch_size = 25
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, LOOK_BACK, 1), return_sequences = True))
    model.add(BatchNormalization())
    model.add(layers.core.Dropout(0.2))
    model.add(LSTM(5,return_sequences=False))
    model.add(layers.core.Dropout(0.2))
    # model.add(Dense(5))
    # model.add(layers.core.Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    lr = 0.005
    decay = 0.95
    nb_epoch = 50
    adam = optimizers.adam(lr=lr)
    # sgd = optimizers.SGD(lr=0.005, clipnorm=0.1)
    model.compile(loss='mean_squared_error', optimizer=adam)

    print "TrainX: ", trainX.shape
    print "TrainY: ", trainY.shape
    print "TestX: ", testX.shape
    print "TestY: ", testY.shape

    for i in range(nb_epoch):
        rd = random.random()
        if rd <0.95:
            # adam.__setattr__('lr', lr)
            adam.lr.set_value(lr)
        else:
            # adam.__setattr__('lr', lr*5)
            adam.lr.set_value(lr*2)

        print 'i: ' , i , ' lr: ' , adam.lr.get_value() #adam.__getattribute__('lr') # adam.lr.get_value()
        model.fit(trainX, trainY, nb_epoch= 1, batch_size=batch_size, verbose=1, shuffle=True, validation_split= 0.15 ) #validation_data=(testX, testY))
        # model.reset_states()
        if i % 5 == 0:
            testPredict = model.predict(testX, batch_size=batch_size, verbose = 1)
            print 'lstm_i: ' , mean_squared_error(testY, testPredict)
        lr *= decay

    # for i in range(100):
    #     model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
    #     model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size, verbose = 1)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size, verbose = 1)


    print 'baseline: ', mean_squared_error(testY, testX[:, -1])
    print 'lstm: ' , mean_squared_error(testY, testPredict)
    print 'avg(abs(.)): ', np.average(np.abs(testY))

def get_train_and_test(dataset, train_size):
    # reshape into X=t and Y=t+1
    # look_back = LOOK_BACK
    # train, test = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

    num_of_months = dataset.shape[0]-1
    num_of_counties = dataset.shape[1]

    print 'num_of_months: ' , num_of_months , ' , num_of_counties: ' , num_of_counties
    print 'start: ' , 0 , ' , end: ', (train_size - LOOK_BACK-1)
    trainX, trainY = create_dataset(dataset,  start= 0 , end = train_size - LOOK_BACK-1, num_of_counties = num_of_counties )
    print 'start: ' , (train_size - LOOK_BACK-1), ' , end: ', ( num_of_months - LOOK_BACK-1)
    testX, testY = create_dataset(dataset,  start= train_size - LOOK_BACK-1 , end = num_of_months - LOOK_BACK-1, num_of_counties = num_of_counties)

    # trainX, trainY = create_dataset(train, look_back)
    # testX, testY = create_dataset(test, look_back)

    print 'data:'
    print trainX[1680,:]
    print trainX[1681,:]
    # trainX = trainX[:2000, :]
    # testX = testX[0:100, :]
    # trainY = trainY[:2000]
    # testY = testY[0:100]

    print "TrainX: ", trainX.shape
    print "TestX: ", testX.shape
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    trainX, trainY = remove_nan(trainX, trainY)
    testX, testY = remove_nan(testX, testY)


    return trainX, trainY, testX, testY

def remove_nan(X, Y):
    print 'remove_nan'
    shape0 = X.shape[0]
    shape1 = X.shape[1]
    for i in reversed(xrange(shape0)):
        remove = 0
        for j in xrange(shape1):
            if X[i,j] is None or math.isnan(X[i,j]):
                remove = 1
        if remove == 1:
            X = np.delete(X, (i), axis=0)
            Y = np.delete(Y, (i), axis=0)
    print X.shape, ' , ', Y.shape
    return X, Y


def build_lstm_on_labels():
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(DB_info.MSP_TABLE) #get_dataframe()
    # dataset = get_county_month(dataframe.values)
    dataset = normalize(get_county_month(dataframe.values), TRAIN_MONTHS)

    print "Dataset shape: ", dataset.shape

    # split into train and test sets
    trainX, trainY, testX, testY = get_train_and_test(dataset, TRAIN_MONTHS)

    # do_pca(trainX, trainY, testX, testY)
    build_LSTM(trainX, trainY, testX, testY)


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)
    build_lstm_on_labels()
    print "--- Completed ---"