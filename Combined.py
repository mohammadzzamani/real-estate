import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from DB_wrapper import DB_wrapper
from keras import optimizers
from keras import layers


import Util
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
COUNTY_COLUMN_NUMBER = 104
#SAF:
# COUNTY_COLUMN_NUMBER = 107
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



# Get only county, month values
def get_county_month(dataset):
    county = dataset[:, COUNTY_COLUMN_NUMBER]
    county = np.reshape(county, (len(county), 1))
    month = dataset[:, INIT_SKIP: INIT_SKIP + MONTH_COLUMNS]

    dataset = np.concatenate((county, month), axis = 1)
    dataset = np.transpose(dataset)
    dataset = dataset.astype('float32')
    return dataset


def build_one_LSTM(trainX, trainY, testX, testY):
    # print 'baseline: ', mean_squared_error(testY, testX[:, -1])
    batch_size = 25
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, LOOK_BACK, 1), return_sequences = True))
    # model.add(BatchNormalization())
    model.add(layers.core.Dropout(0.2))
    model.add(LSTM(5,return_sequences=False))
    model.add(layers.core.Dropout(0.2))

    return model


def build_LSTM(trainX, trainY, testX, testY):
    print 'baseline: ', mean_squared_error(testY, testX[:, -1])
    batch_size = 25
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, LOOK_BACK, 1), return_sequences = True))
    # model.add(BatchNormalization())
    model.add(layers.core.Dropout(0.2))
    model.add(LSTM(5,return_sequences=False))
    model.add(layers.core.Dropout(0.2))
    # model.add(Dense(5))
    # model.add(layers.core.Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    lr = 0.005
    decay = 0.95
    nb_epoch = 100
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

    # trainX = trainX[:2000, :]
    # testX = testX[0:100, :]
    # trainY = trainY[:2000]
    # testY = testY[0:100]

    print "TrainX: ", trainX.shape
    print "TestX: ", testX.shape
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    trainX, trainY = Util.remove_nan(trainX, trainY)
    testX, testY = Util.remove_nan(testX, testY)

    return trainX, trainY, testX, testY

def build_lstm_on_labels():
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(DB_info.MSP_FEATURES) #get_dataframe()
    # dataset = get_county_month(dataframe.values)

    ####### do the reshape
    data = dataframe.values

    lstm = []
    for i in xrange(data.shape[0]):

        dataset = Util.normalize_min_max(get_county_month(dataframe.values), TRAIN_MONTHS)

        print "Dataset shape: ", dataset.shape

        # split into train and test sets
        trainX, trainY, testX, testY = get_train_and_test(dataset, TRAIN_MONTHS)

        lstm.append(build_one_LSTM(trainX, trainY, testX, testY))


def Neural_Net(models):

    model = Sequential()
    model.add(Merge(models, mode = 'concat'))
    model.add(Dense(output_dim = 50, input_dim = len(models), init = 'normal', activation = 'sigmoid'))
    model.add(Dense(output_dim = 10, init = 'normal', activation = 'tanh'))
    model.add(Dense(output_dim = 1, init = 'normal'))

    # Compile model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    model.fit(xTrain, yTrain, nb_epoch = 10, batch_size = 100, validation_split = 0.1)

    score = model.evaluate(xTest, yTest, batch_size = 100)
    prediction = model.predict(xTest, batch_size = 100, verbose = 1)

    result = [(yTest[i], prediction[i][0]) for i in xrange(0, 30)]


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)
    build_lstm_on_labels()
    print "--- Completed ---"