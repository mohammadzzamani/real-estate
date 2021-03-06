from DB_wrapper import DB_wrapper
import pandas as pd
import random
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import Util
import math
import warnings

import DB_info

# Initially 8 columns
# columns from 2008/07 to 2016/09 = 6 + 84 + 9 = 99 columns
# We need 45 columns between 2011/11 and 2015/07
# So skip 8 initial columns + columns from 2008/07 to 2011/10 = 8 + 6 + 12*2 + 10 = 48 columns
INIT_SKIP = 48
MONTH_COLUMNS = 45
TRAIN_MONTHS = 36
# TEST_MONTHS = 9

# After all months, comes county column which is column number 99 + 8 = 107
#IP:
#COUNTY_COLUMN_NUMBER = 113
#MSP:
COUNTY_COLUMN_NUMBER = 104
#SAF:
#COUNTY_COLUMN_NUMBER = 107
LOOK_BACK = 30

def remove_nan_arima(X):
    print 'remove_nan_arima'
    print 'Before: ', X.shape
    shape0 = X.shape[0]
    shape1 = X.shape[1]
    for i in reversed(xrange(shape0)):
        remove = 0
    	for j in xrange(shape1):
    	    if X[i,j] is None or math.isnan(X[i,j]):
                remove = 1
                break

        if remove == 1:
            X = np.delete(X, (i), axis=0)

    print 'Remove Nan - After: ', X.shape
    return X

def get_mean(dataset, train_size):
    previous_month = list()
    current_month = list()

    for column in xrange(dataset.shape[1]):
        train = dataset[: train_size, column]

        for i in range(train_size - 2):
            p = float(train[i])
            c = float(train[i + 1])

            #if p is None or c is None or math.isnan(p) or math.isnan(c):
            #    continue

            previous_month.append(p)
            current_month.append(c)

    mean = np.mean(current_month) - np.mean(previous_month)
    print "Previous month mean: ", np.mean(previous_month)
    print "Current month mean: ", np.mean(current_month)
    return mean
    

def compute_baseline(dataset, train_size):
    previous_month = list()
    current_month = list()

    for column in xrange(dataset.shape[1]):
        test = dataset[train_size:, column]

        for i in range(test.shape[0]):
            previous = float(dataset[train_size + i - 1, column])
            current = float(test[i])

            #if previous is None or current is None or math.isnan(previous) or math.isnan(current):
            #    continue

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


with warnings.catch_warnings():
    print 'warning here'
    warnings.filterwarnings("ignore")
    # Line that is not converging


def build_ARIMA(dataset, train_size, counties, order = (3, 0 , 2)):
    # Write ARIMA code here
    small_order = (3, 0 , 2)
    smaller_order = (2, 0 , 1)
    print 'order: ' , order , ' , ', small_order
    prediction_dataframe = pd.DataFrame()
    predictions = list()
    observed_labels = list()
    previous_labels = list()

    # Predict from 35th to 45th month for each county
    result_1 = 0
    result_0 = 0
    for column in xrange(dataset.shape[1]):
        #print "Train Size: ", train_size, ", Column: ", column
        train, test = dataset[: train_size, column], dataset[train_size: , column]
        history = train.tolist()

        for t in xrange(test.shape[0]):
            try:
                model = ARIMA(history, order = order )
                model_fit = model.fit(disp = 0, maxiter = 50)
                if not model_fit.mle_retvals['converged']:
                    print 'not converged first'
                    raise Exception
            except:
                try:
                    model_small = ARIMA(history, order = small_order )
                    model_fit = model_small.fit(disp = 0, maxiter = 100)
                    if not model_fit.mle_retvals['converged']:
                        print 'not converged second'
                        raise  Exception
                except:
                    try:
                        model_smaller = ARIMA(history, order = smaller_order )
                        model_fit = model_smaller.fit(disp = 0, maxiter = 100)
                        if not model_fit.mle_retvals['converged']:
                            print 'not converged third'
                            continue
                    except:
                        continue


                output = model_fit.forecast()
                yHat = output[0]
                previous = dataset[train_size + t - 1, column]

                history.append(test[t])
                if yHat is not None and math.isnan(yHat) == False:
                    predictions.append(yHat)
                    observed_labels.append(test[t])
                    previous_labels.append(previous)

                    if np.sign(yHat - previous) == np.sign(test[t] - previous):
                        res = 1
                        result_1 +=1
                    else:
                        res = 0
                        result_0+=1
                    print ' ... ' , column , ' , ', counties[column] , ' , ',  t , ' : ' , previous , ' , ', yHat , ' , ', test[t] , ' , ', res , ' , ', result_0 , ' , ', result_1
                    cnty_month = str(counties[column]) + '_' + str(t)
                    prediction_dataframe = prediction_dataframe.append({'cnty_month': cnty_month, 'arima': yHat}, ignore_index = True)

        if column % 15 == 0 and column <> 0:
            pr = np.asarray(predictions)
            pl = np.asarray(previous_labels)
            ol = np.asarray(observed_labels)

            pr = pr.reshape(pr.shape[0])
            pl = pl.reshape(pl.shape[0])
            ol = ol.reshape(ol.shape[0])
            
            print "County: ", column
            error = mean_squared_error(ol, pr)
            print ('Test(MSE): %.6f' % error)
            error = mean_absolute_error(ol, pr)
            print ('Test(MAE): %.6f' % error)
            print 'Test Accuracy: ' , sum(1 for x, y in zip(np.sign(pr - pl), np.sign(ol - pl)) if x == y) / float(len(ol))

            for i in xrange(10):
                print "Predicted: ", predictions[len(predictions) - i - 1], ", Expected: ", observed_labels[len(observed_labels) - i - 1]
            
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

    print 'Test Accuracy: ' , sum(1 for x, y in zip(np.sign(predictions - previous_labels), np.sign(observed_labels - previous_labels)) if x == y) / float(len(observed_labels))
    return prediction_dataframe


def create_dataframe(counties, predictions):
    res = pd.DataFrame()
    idx = 0
    print "Predictions: ", len(predictions)

    for county in counties:
        for i in range(TRAIN_MONTHS, MONTH_COLUMNS):
            cnty_month = str(county) + '_' + str(i)
            res = res.append({'cnty_month': cnty_month, 'arima': predictions[idx]}, ignore_index = True)
            idx += 1

    return res


def build_arima_on_labels(table = DB_info.MSP_LIMITED_TABLE, county_column_number = COUNTY_COLUMN_NUMBER, train_month = TRAIN_MONTHS, order = ( 3, 0 , 2) ):
    db_wrapper = DB_wrapper()
    dataframe = db_wrapper.retrieve_data(table )
    dataset = get_county_month(dataframe.values)
    COUNTY_COLUMN_NUMBER = county_column_number
    print "Column: ", COUNTY_COLUMN_NUMBER
    print "Dataset shape: ", dataset.shape
    dataset = Util.normalize_each_county_ndarry(np.transpose(dataset[1:, :]), MONTH_COLUMNS, dataset.shape[1])
    dataset = np.transpose(remove_nan_arima(dataset))
    
    TRAIN_MONTHS = train_month
    compute_baseline(dataset, TRAIN_MONTHS)
    counties = dataframe.values[:, COUNTY_COLUMN_NUMBER].tolist()
    prediction_dataframe = build_ARIMA(dataset, TRAIN_MONTHS, counties, order)

    return prediction_dataframe



if __name__ == "__main__":
 # fix random seed for reproducibility
    np.random.seed(7)
    build_arima_on_labels()
    print "--- Completed ---"
