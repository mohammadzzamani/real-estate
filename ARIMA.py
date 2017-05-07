from DB_wrapper import DB_wrapper
import random
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

        if remove == 1:
            X = np.delete(X, (i), axis=0)

    return X

def get_mean(dataset, train_size):
    previous_month = list()
    current_month = list()

    for column in xrange(dataset.shape[1]):
        train = dataset[: train_size, column]

        for i in range(train_size - 2):
            p = float(train[i + 1])
            c = float(dataset[i + 2, column])

            if p is None or c is None or math.isnan(p) or math.isnan(c):
                continue

            previous_month.append(p)
            current_month.append(c)

    mean = np.mean(previous_month) - np.mean(current_month)
    print "Previous month mean: ", np.mean(previous_month)
    print "Current month mean: ", np.mean(current_month)
    return mean
    

def compute_baseline(dataset, train_size):
    previous_month = list()
    current_month = list()

    for column in xrange(dataset.shape[0]):
        test = dataset[train_size:, column]

        for i in range(test.shape[0]):
            previous = float(dataset[train_size + i - 1, column])
            current = float(test[i])

            if previous is None or current is None or math.isnan(previous) or math.isnan(current):
                continue

            previous_month.append(previous)
            current_month.append(current)
            #print "Previous: ", previous, ", Current: ", current
    
    differences = [current_month[i] - previous_month[i] for i in xrange(len(current_month))]
    mean = get_mean(dataset, train_size)
    m_month = [mean + i for i in previous_month]

    print "Mean: ", mean
    print "Baseline1 (MAE): ", mean_absolute_error(previous_month, current_month)
    print "Baseline1 (MSE): ", mean_squared_error(previous_month, current_month)
    print "Baseline2 (MAE): ", mean_absolute_error(m_month, current_month)
    print "Baseline2 (MSE): ", mean_squared_error(m_month, current_month)
    print 'Test Accuracy: ' , sum(1 for x,y in zip(np.sign([mean for i in current_month]), np.sign(differences)) if x == y) / float(len(current_month))


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
    previous_labels = list()

    # Predict from 35th to 45th month for each county
    for column in xrange(dataset.shape[0]):
        #print "Train Size: ", train_size, ", Column: ", column
        train, test = dataset[: train_size, column], dataset[train_size: , column]
        history = []
        for i in range(len(train)):
            if train[i] is None or math.isnan(train[i]):
                continue

            history.append(train[i])

        print "Test shape: ", test.shape
        for t in xrange(test.shape[0]):
            try:
                model = ARIMA(history, order = (3, 0, 0))
                model_fit = model.fit(disp = 0)
                output = model_fit.forecast()
                yHat = output[0]
                previous = dataset[train_size + t - 1, column]

                if (test[t] != None and math.isnan(test[t]) == False and previous != None and math.isnan(previous) == False):
                    history.append(test[t])
                    predictions.append(yHat)
                    observed_labels.append(test[t])
                    previous_labels.append(previous)
                    print ("(%d. %d) - Predicted: %f, Expected: %f" % (column, t, yHat, test[t]))
                else:
                    print "Skipped: ", column, ", ", t
            except:
                pass

    print "---------- ARIMA ----------"
    error = mean_squared_error(observed_labels, predictions)
    print ('Test(MSE): %.6f' % error)
    error = mean_absolute_error(observed_labels, predictions)
    print ('Test(MAE): %.6f' % error)
    #print 'Test Accuracy: ' , sum(1 for x,y in zip(predictions, observed_labels) if x == y) / float(len(observed_labels))

    predictions = np.asarray(predictions)
    previous_labels = np.asarray(previous_labels)
    observed_labels = np.asarray(observed_labels)

    predictions = predictions.reshape(predictions.shape[0])
    previous_labels = previous_labels.reshape(previous_labels.shape[0])
    observed_labels = observed_labels.reshape(observed_labels.shape[0])

    print 'Test Accuracy: ' , sum(1 for x, y in zip(np.sign(predictions - previous_labels), np.sign(observed_labels, previous_labels)) if x == y) / float(len(observed_labels))

def build_arima_on_labels():
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(DB_info.MSP_LIMITED_TABLE) #get_dataframe()
    # dataset = get_county_month(dataframe.values)
    dataset = Util.normalize_min_max(get_county_month(dataframe.values), TRAIN_MONTHS)

    print "Dataset shape: ", dataset.shape
    #dataset = remove_nan_arima(dataset)

    # Util.do_pca(trainX, trainY, testX, testY)
    compute_baseline(dataset, TRAIN_MONTHS)
    build_ARIMA(dataset, TRAIN_MONTHS)


if __name__ == "__main__":
 # fix random seed for reproducibility
    np.random.seed(7)
    build_arima_on_labels()
    print "--- Completed ---"
