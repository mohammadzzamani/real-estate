from DB_wrapper import DB_wrapper
import random
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import Util
import math

# from sqlalchemy import create_engine
# from sqlalchemy.engine.url import URL

import DB_info

# Initially 8 columns
# columns from 2008/07 to 2016/09 = 6 + 84 + 9 = 99 columns
# We need 45 columns between 2011/11 and 2015/07
# So skip 8 initial columns + columns from 2008/07 to 2011/10 = 8 + 6 + 12*2 + 10 = 48 columns
INIT_SKIP = 48
MONTH_COLUMNS = 45
TRAIN_MONTHS = 35
# TEST_MONTHS = 9

# After all months, comes county column which is column number 99 + 8 = 107
#IP:
# COUNTY_COLUMN_NUMBER = 113
#MSP:
# COUNTY_COLUMN_NUMBER = 104
#SAF:
COUNTY_COLUMN_NUMBER = 107
LOOK_BACK = 30

def remove_nan_arima(X):
    print 'remove_nan_arima'
    shape0 = X.shape[0]
    shape1 = X.shape[1]
    for i in reversed(xrange(shape0)):
        remove = 0
    	for j in xrange(shape1):
    	    if X[i,j] is None or math.isnan(X[i,j]):
                remove = 1
                print "Here ", i
                break

        if remove == 1:
            X = np.delete(X, (i), axis=0)

    return X


# Get only county, month values
def get_county_month(dataset):
    county = dataset[:, COUNTY_COLUMN_NUMBER]
    county = np.reshape(county, (len(county), 1))
    month = dataset[:, INIT_SKIP: INIT_SKIP + MONTH_COLUMNS]

    dataset = np.concatenate((county, month), axis = 1)
    dataset = np.transpose(dataset)
    dataset = dataset.astype('float32')
    return dataset


def build_ARIMA(dataset, train_size):
    # Write ARIMA code here
    predictions = list()
    observed_labels = list()

    # Predict from 35th to 45th month for each county
    for column in xrange(dataset.shape[1]):
        #print "Train Size: ", train_size, ", Column: ", column
        train, test = dataset[: train_size, column], dataset[train_size: , column]
        history = []
        for i in range(len(train)):
            if train[i] is None or math.isnan(train[i]):
                continue

            history.append(train[i])

        #print "Test: ", test
        #print "Dataset: ", dataset.shape

        #exit()
        # Check for each month after 35th month in Test
        for t in xrange(test.shape[0]):
            try:
                model = ARIMA(history, order = (5, 0, 0))
                model_fit = model.fit(disp = 0)
                output = model_fit.forecast()
                yHat = output[0]

                if (test[t] != None and math.isnan(test[t]) == False):
                    history.append(test[t])
                    predictions.append(yHat)
                    observed_labels.append(test[t])
                    print ("(%d. %d) - Predicted: %f, Expected: %f" % (county, t, yHat, test[t]))
                else:
                    print "Skipped: ", column, ", ", t
            except:
                pass

    error = mean_squared_error(observed_labels, predictions)
    print ('Test MSE: %.6f' % error)


def build_arima_on_labels():
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(DB_info.SAF_TABLE) #get_dataframe()
    # dataset = get_county_month(dataframe.values)
    dataset = Util.normalize_min_max(get_county_month(dataframe.values), TRAIN_MONTHS)

    print "Dataset shape: ", dataset.shape
    #dataset = remove_nan_arima(dataset)

    # Util.do_pca(trainX, trainY, testX, testY)
    build_ARIMA(dataset, TRAIN_MONTHS)


if __name__ == "__main__":
 # fix random seed for reproducibility
    np.random.seed(7)
    build_arima_on_labels()
    print "--- Completed ---"
