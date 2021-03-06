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
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeCV

import ARIMA


# After all months, comes county column which is column number 99 + 8 = 107
#IP:
#COUNTY_COLUMN_NUMBER = 113
#MSP:
COUNTY_COLUMN_NUMBER = 104
#SAF:
#COUNTY_COLUMN_NUMBER = 107

# DATABASE = 'mztwitter'
# TRAIN_TABLE_NAME = 'NLP_features_msp'
# TEST_TABLE_NAME = 'NLP_test_features_saf'
ID_SIZE = 0

# Change this depending on whatever is the number of features
# in the dataframe.
NUM_FEATURES = 32
TOTAL_MONTHS = 45
MONTH = 'month'

class NeuralNetwork_:

    # Returns the ID columns as numpy ndarray
    def get_ids(self, dataframe):
        return dataframe.ix[:, 0: ID_SIZE].values


    def prev_cnty_month ( self, cnty_month):
        [cnty , month ]   = cnty_month.split('_')
        month = int(month)
        prev_index = str(cnty)+'_'+str(month-1)
        return prev_index


    def merge_with_prev(self, df ):

        df_prev = df.copy()
        df_prev.index = df_prev.index.map(self.prev_cnty_month)
        new_df = df_prev.join(df,  how='inner', lsuffix='_prev')
        return new_df


    def split_train_test(self, new_df , train_month ):

        test_df = new_df[new_df.month > train_month]
        train_df = new_df[new_df.month <= train_month]
        print new_df.columns

        train_df.drop('cnty', axis=1, inplace=True)
        train_df.drop('month', axis=1, inplace=True)
        train_df.drop('cnty_prev', axis=1, inplace=True)
        train_df.drop('month_prev', axis=1, inplace=True)

        test_df.drop('cnty', axis=1, inplace=True)
        test_df.drop('month', axis=1, inplace=True)
        test_df.drop('cnty_prev', axis=1, inplace=True)
        test_df.drop('month_prev', axis=1, inplace=True)

        print 'train_df.shape: ' , train_df.shape
        print 'test_df.shape: ' ,test_df.shape

        return train_df, test_df


    def baseline_model(self,xTrain, xTest, yTrain, yTest, yPrevTest, yPrevTrain):
        print 'neural network model'

        print 'yPrevTest'
        print yPrevTest
        model = Sequential()
        model.add(Dense(20, input_dim=len(xTrain[0]) , init='normal', activation='linear'))
        model.add(Dense(1, init='normal'))

        lr = 0.05
        decay = 0.975
        adam = optimizers.adam(lr = lr, decay = decay)
        model.compile(loss = 'mean_absolute_error', optimizer = adam)
        nb_epochs = 150
        for i in xrange(nb_epochs):
            lr = lr * decay
            adam.lr.set_value(lr)
            model.fit(xTrain, yTrain, nb_epoch = 1, batch_size = 5000, shuffle = True, validation_split = 0.1, verbose = 1)

            if i %10 == 0:
                score = model.evaluate(xTest, yTest, batch_size = 5000)
                print 'score: ' , score

                testPredict = model.predict(xTest, verbose = 1)
                print 'Neural Network_i: ', i , ' , ' ,   mean_squared_error(yTest, testPredict)
                print 'Neural Network_i: ', i , ' , ', mean_absolute_error(yTest, testPredict)

                testPredict = testPredict.reshape(testPredict.shape[0])

                print 'testPredict'
                print testPredict[:10]
                print 'yTest'
                print yTest[:10]
                print ' yPrevTest'
                print yPrevTest[:10]
                print ' test accuracy: ' , sum(1 for x,y in zip(np.sign(testPredict - yPrevTest),np.sign(yTest- yPrevTest)) if x == y) / float(len(yTest))

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
        model.add(Dense(output_dim = 5, init='normal' , activation = 'relu'))
        model.add(Dense(output_dim = 1, init='normal' , activation = 'linear'))

        lr = 1.0

        adam = optimizers.adam(lr = lr)
        model.compile(loss = 'mean_squared_error', optimizer = adam)
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

    def compute_baseline(self, train_df ,  test_df):
        print 'compute_baseline:'

        mean = np.mean(train_df.label) - np.mean(train_df.label_prev)

        mean_df = test_df.copy()
        print 'avg: ' , mean
        mean_df['avg'] = mean

        diff  = mean_df['label'] - mean_df['label_prev']

        print 'baseline1 (MAE): ' , mean_absolute_error(mean_df.label_prev, mean_df.label)
        print 'baseline1 (MSE): ', mean_squared_error(mean_df.label_prev, mean_df.label)

        print mean_df.avg.shape
        print diff[:10]
        print len(diff)
        print diff.shape
        print 'baseline2 (MAE):  ', mean_absolute_error(mean_df.avg, diff)
        print 'baseline2 (MSE): ', mean_squared_error(mean_df.avg, diff)
        print ' test accuracy: ' , sum(1 for x,y in zip(np.sign(mean_df.avg),np.sign(diff)) if x == y) / float(len(mean_df.label))


    def linear_model(self, train_set, test_set):
        print ' ridge '

        xTrain = train_set.ix[:, :-1].values
        yTrain = train_set.ix[:,-1].values
        xTest = test_set.ix[:, :-1].values
        yTest = test_set.ix[:,-1].values

        yPrevTest = test_set.ix[:, NUM_FEATURES].values
        yPrevTrain = train_set.ix[:,NUM_FEATURES].values

        cvParams = {'ridgecv': [{'alphas': np.array([1, .1, .01, .001, .0001, 10, 100, 1000, 10000, 100000, 100000, 1000000])}]}
        ridge = RidgeCV()
        ridge.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in cvParams['ridgecv'][0].iteritems()))
        ridge.fit(xTrain, yTrain)
        ridge_pred_test = ridge.predict(xTest)
        ridge_pred_train = ridge.predict(xTrain)

        print 'Result_test: ', mean_squared_error(yTest, ridge_pred_test)
        print 'Result_train: ', mean_squared_error(yTrain, ridge_pred_train)

        print 'Result_test: ', mean_absolute_error(yTest, ridge_pred_test)
        print 'Result_train: ', mean_absolute_error(yTrain, ridge_pred_train)

        print ' ridge test accuracy: ' , sum(1 for x,y in zip(np.sign(ridge_pred_test - yPrevTest),np.sign(yTest - yPrevTest)) if x == y) / float(len(yTest))
        print ' ridge train accuracy: ' , sum(1 for x,y in zip(np.sign(ridge_pred_train - yPrevTrain),np.sign(yTrain - yPrevTrain)) if x == y) / float(len(yTrain))

        ridge_coef = ridge.coef_
        print 'ridge_coef: '
        print ridge_coef

        print 'alpha: '
        print ridge.alpha_

        print 'linear regression'
        lr = linear_model.LinearRegression()
        lr.fit(xTrain, yTrain)
        lr_pred_test = lr.predict(xTest)
        lr_pred_train = lr.predict(xTrain)

        print 'Result_test: ', mean_squared_error(yTest, lr_pred_test)
        print 'Result_train: ', mean_squared_error(yTrain, lr_pred_train)

        print 'Result_test: ', mean_absolute_error(yTest, lr_pred_test)
        print 'Result_train: ', mean_absolute_error(yTrain, lr_pred_train)

        print ' lr test accuracy: ' , sum(1 for x,y in zip(np.sign(lr_pred_test - yPrevTest),np.sign(yTest - yPrevTest)) if x == y) / float(len(yTest))
        print ' lr train accuracy: ' , sum(1 for x,y in zip(np.sign(lr_pred_train - yPrevTrain),np.sign(yTrain - yPrevTrain)) if x == y) / float(len(yTrain))

        print  'lr.coef_: '
        print lr.coef_


    def linear_classifier(self, type, train_set, test_set):
        print ' linear_classifier '

        xTrain = train_set.ix[:, :-1].values
        yTrain = train_set.ix[:,-1].values
        xTest = test_set.ix[:, :-1].values
        yTest = test_set.ix[:,-1].values

        yPrevTest = test_set.ix[:,  NUM_FEATURES].values
        yPrevTrain = train_set.ix[:,  NUM_FEATURES].values

        yTest = np.sign(yTest - yPrevTest)
        yTrain = np.sign(yTrain - yPrevTrain)

        if type == 'SGDClassifier':
            clf = linear_model.SGDClassifier()
        elif type == 'poly':
            clf = SVR(kernel='poly', C=1e3, degree=2)

        clf.fit(xTrain, yTrain)
        clf_pred_test = clf.predict(xTest)
        clf_pred_train = clf.predict(xTrain)

        print 'examples:'
        print yTest[:10]
        print clf_pred_test[:10]

        print 'Result_test: ', mean_squared_error(yTest, clf_pred_test)
        print 'Result_train: ', mean_squared_error(yTrain, clf_pred_train)

        print 'Result_test: ', mean_absolute_error(yTest, clf_pred_test)
        print 'Result_train: ', mean_absolute_error(yTrain, clf_pred_train)

        print ' clf test accuracy: ' , sum(1 for x,y in zip(clf_pred_test,yTest ) if x == y) / float(len(yTest))
        print ' clf train accuracy: ' , sum(1 for x,y in zip(clf_pred_train,yTrain ) if x == y) / float(len(yTrain))

    def __init__(self):
        print "-- Created NeuralNetwork Object --"


if __name__ == "__main__":
    db_wrapper = DB_wrapper()
    dataframe_train = db_wrapper.retrieve_data(DB_info.FEATURE_TABLE) #get_dataframe(DATABASE, TRAIN_TABLE_NAME)
    dataframe_train = dataframe_train.set_index('cnty_month')

    print 'dataframe_train before: ' , dataframe_train.shape
    # dataframe_train = dataframe_train.dropna(how='any')
    dataframe_train  = dataframe_train.dropna(subset=['label'], how = 'all')
    dataframe_train = dataframe_train[np.isfinite(dataframe_train['label'])]
    print 'dataframe_train after: ' , dataframe_train.shape

    dataframe_train = Util.normalize_each_county(dataframe_train, TOTAL_MONTHS,  NUM_FEATURES)

    Network = NeuralNetwork_()
    new_dataframe = Network.merge_with_prev(dataframe_train )

    # new_dataframe = arima_df.join(new_dataframe,  how='inner')
    print 'new_dataframe.columns: ' , new_dataframe.columns
    print 'new_dataframe.shape: ', new_dataframe.shape

    [train_set , test_set] = Network.split_train_test(new_dataframe,  0.8 * TOTAL_MONTHS)

    print 'dataframe_train after adding prev_data: ' , train_set.shape , ' , ', test_set.shape

    xTrain = train_set.ix[:, :-1].values
    yTrain = train_set.ix[:,-1].values
    xTest = test_set.ix[:, :-1].values
    yTest = test_set.ix[:,-1].values

    print 'columns: ' , test_set.columns
    yPrevTest = test_set.ix[:,NUM_FEATURES].values
    print 'yPrevTest:::::::::::::::::::'
    print yPrevTest
    yPrevTrain = train_set.ix[:,NUM_FEATURES].values

    Network.compute_baseline( train_set, test_set)

    #linear regression
    Network.linear_model( train_set, test_set)
    Network.linear_classifier('SGDClassifier', train_set, test_set)

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