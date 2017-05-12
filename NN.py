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
NUM_FEATURES = 34
TOTAL_MONTHS = 45

MONTH = 'month'

class NN:

    # Returns the ID columns as numpy ndarray
    def get_ids(self, dataframe):
        return dataframe.ix[:, 0: ID_SIZE].values


    def prev_cnty_month ( self, cnty_month):
        [cnty , month ]   = cnty_month.split('_')
        month = int(month)
        prev_index = str(cnty)+'_'+str(month+1)
        return prev_index


    def merge_with_prev(self, df ):

        df_prev = df.copy()
        df_prev.index = df_prev.index.map(self.prev_cnty_month)
        new_df = df_prev.join(df,  how='inner', lsuffix='_prev')
        df_prev1 = df_prev.copy()
        df_prev1.index = df_prev1.index.map(self.prev_cnty_month)
        new_df1 = df_prev1.join(new_df,  how='inner', lsuffix='_prev_2')

        return new_df1




    def split_train_test(self, new_df , train_month ):

        test_df = new_df[new_df.month > train_month]
        train_df = new_df[new_df.month <= train_month]
        print new_df.columns

        train_df.drop('cnty', axis=1, inplace=True)
        # train_df.drop('month', axis=1, inplace=True)
        # train_df.drop('cnty_prev', axis=1, inplace=True)
        # train_df.drop('month_prev', axis=1, inplace=True)
        # train_df.drop('cnty_prev_2', axis=1, inplace=True)
        # train_df.drop('month_prev_2', axis=1, inplace=True)

        test_df.drop('cnty', axis=1, inplace=True)
        # test_df.drop('month', axis=1, inplace=True)
        # test_df.drop('cnty_prev', axis=1, inplace=True)
        # test_df.drop('month_prev', axis=1, inplace=True)
        # test_df.drop('cnty_prev_2', axis=1, inplace=True)
        # test_df.drop('month_prev_2', axis=1, inplace=True)

        print 'train_df.shape: ' , train_df.shape
        print 'test_df.shape: ' ,test_df.shape

        return train_df, test_df




    def neural_net(self,train_set, test_set):
        print  '<<<<<<<<<<<<<<<<<<<<<< neural_net  >>>>>>>>>>>>>>>>>>>>>>>> '
        xTrain, xTest, yTrain, yTest, yPrevTest, yPrevTrain, yPrevIndex = Network.prepare_data(train_set, test_set)

        print 'yPrevTest'
        print yPrevTest
        # yTest = np.sign(yTest - yPrevTest) #.
        # yTrain = np.sign(yTrain - yPrevTrain) #.
        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=len(xTrain[0]) , init='normal', activation='relu'))  #*
        # model.add(Dense(20, input_dim=len(xTrain[0]) , init='normal', activation='tanh'))  #.
        # model.add(layers.core.Dropout(0.2))
        # model.add(Dense(output_dim = 30, init='normal' , activation = 'relu'))
        # model.add(layers.core.Dropout(0.2))
        # model.add(Dense(output_dim = 10, init='normal' , activation = 'relu'))
        # model.add(layers.core.Dropout(0.1))
        model.add(Dense(output_dim = 5, init='normal' , activation = 'linear'))  #*
        # model.add(Dense(output_dim = 5, init='normal' , activation = 'linear'))  #.
        # model.add(Dense(1, init='normal', activation= 'softmax')) #.
        model.add(Dense(1, init='normal'))
        # Compile model

        lr = 0.05
        decay = 0.975
        adam = optimizers.adam(lr = lr, decay = decay)
        # model.compile(loss = 'binary_crossentropy', optimizer = adam) #.
        model.compile(loss = 'mean_absolute_error', optimizer = adam)  #*
        nb_epochs = 500
        for i in xrange(nb_epochs):

            adam.lr.set_value(lr)
            model.fit(xTrain, yTrain, nb_epoch = 1, batch_size = 5000, shuffle = True, validation_split = 0.2, verbose = 1)

            if i %10 == 0:
                score = model.evaluate(xTest, yTest, batch_size = 5000)
                print 'score: ' , score
                lr = lr * decay
                testPredict = model.predict(xTest, verbose = 1)
                testPred =  [ max( min(val, 150)  , -25 ) for val in testPredict]
                # testPredict = testPred
                print 'Neural Network_i: ', i , ' , ' ,  lr , ' , ' ,   mean_squared_error(yTest, testPredict)
                print 'Neural Network_i: ', i , ' , ', mean_absolute_error(yTest, testPredict)
                print 'Neural Network_i: ', i , ' , ' ,  lr , ' , ' ,   mean_squared_error(yTest, testPred)
                print 'Neural Network_i: ', i , ' , ', mean_absolute_error(yTest, testPred)
                # print 'Neural Network accuracy: ' , sum(1 for x,y in zip(testPredict,yTest ) if x == y) / float(len(yTest)) #.

                testPredict = testPredict.reshape(testPredict.shape[0])  #*
                print ' test accuracy: ' , sum(1 for x,y in zip(np.sign(testPredict - yPrevTest),np.sign(yTest- yPrevTest)) if x == y) / float(len(yTest))  #*
                # print 'yPrevTest'
                # print yPrevTest
                # x1 = np.sign(testPredict - yPrevTest)
                # x2 = np.sign(yTest- yPrevTest)

                # a = []
                # a.append(yPrevTest.tolist())
                # a.append(yTest)
                # a.append(testPred)
                # a  = np.transpose(a)
                # print a[:25,:]
                # print a[25:50,:]

                print 'testPredict'
                print testPredict[:10]
                print 'yTest'
                print yTest[:10]
                print ' yPrevTest'
                print yPrevTest[:10]

                # print 's1:'
                # s1 = np.sign(testPredict - yPrevTest)
                # print s1[:100]
                # print 's2'
                # s2 = np.sign(yTest- yPrevTest)
                # print s2[:100]

                # print ' accuracy: ' , mean_absolute_error(x1, x2)

                # print ' lr test accuracy: ' , sum(1 for x,y in zip(np.sign(lr_pred_test - yPrevTest),np.sign(yTest - yPrevTest)) if x == y) / float(len(yTest))

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




    def compute_baseline(self, train_df ,  test_df):
        print ' <<<<<<<<<<<<<<<<<<<<<< compute_baseline  >>>>>>>>>>>>>>>>>>>>>>>>'

        mean = np.mean(train_df.label) - np.mean(train_df.label_prev)
        print 'average: ' , mean

        mean_df = test_df.copy()
        mean_df['avg_dif'] = mean


        print 'baseline1 (MAE): ' , mean_absolute_error(mean_df.label_prev, mean_df.label)
        print 'baseline1 (MSE): ', mean_squared_error(mean_df.label_prev, mean_df.label)

        mean_df['dif']  = mean_df['label'] - mean_df['label_prev']
        print 'mean_df.avg_dif.shape: ' , mean_df.avg_dif.shape
        print 'mean_df.dif.shape: ' , mean_df.dif.shape

        print 'baseline2 (MAE):  ', mean_absolute_error(mean_df.avg_dif, mean_df.dif)
        print 'baseline2 (MSE): ', mean_squared_error(mean_df.avg_dif, mean_df.dif)


        print 'baseline2 test accuracy: ' , sum(1 for x,y in zip(np.sign(mean_df.avg_dif),np.sign(mean_df.dif)) if x == y) / float(len(mean_df.label))

        print 'baseline3 (MAE): ' , mean_absolute_error(mean_df.label_prev_2, mean_df.label)
        print 'baseline3 (MSE): ', mean_squared_error(mean_df.label_prev_2, mean_df.label)


    def linear_model(self, train_set, test_set, type = 'ridge_regression'):
        print '          <<<<<<<<<<<<<<<<<<<<<< linear_model  >>>>>>>>>>>>>>>>>>>>>>>> '


        xTrain, xTest, yTrain, yTest, yPrevTest, yPrevTrain, yPrevIndex = self.prepare_data(train_set,test_set)



        if type == 'ridge_regression':
            print '      <<< ridge-regression >>> '
            cvParams = {'ridgecv': [{'alphas': np.array([1, .1, .01, .001, .0001, 10, 100, 1000, 10000, 100000, 100000, 1000000])}]}
            model = RidgeCV()
            model.set_params(**dict((k, v[0] if isinstance(v, list) else v) for k,v in cvParams['ridgecv'][0].iteritems()))
        else:
            print '      <<< linear_regression >>>'
            model = linear_model.LinearRegression()


        model.fit(xTrain, yTrain)
        pred_test = model.predict(xTest)
        pred_train = model.predict(xTrain)

        print 'test MSE: ', mean_squared_error(yTest, pred_test)
        print 'train MSE: ', mean_squared_error(yTrain, pred_train)

        print 'test MAE: ', mean_absolute_error(yTest, pred_test)
        print 'train MAE: ', mean_absolute_error(yTrain, pred_train)


        print 'test accuracy: ' , sum(1 for x,y in zip(np.sign(pred_test - yPrevTest),np.sign(yTest - yPrevTest)) if x == y) / float(len(yTest))
        print 'train accuracy: ' , sum(1 for x,y in zip(np.sign(pred_train - yPrevTrain),np.sign(yTrain - yPrevTrain)) if x == y) / float(len(yTrain))

        coef = model.coef_
        print 'coef: '
        print coef

        if type == 'ridge_regression':
            print 'best alpha: '
            print model.alpha_



    def linear_classifier(self, type, train_set, test_set):
        print '      <<<<<<<<<<<<<<<<<<<<<< linear_classifier  >>>>>>>>>>>>>>>>>>>>>>>>'

        xTrain, xTest, yTrain, yTest, yPrevTest, yPrevTrain, yPrevIndex = self.prepare_data(train_set, test_set)

        yTest = np.sign(yTest - yPrevTest)
        yTrain = np.sign(yTrain - yPrevTrain)

        # lr = linear_model.LinearRegression()
        if type == 'SGDClassifier':
            clf = linear_model.SGDClassifier()
        elif type == 'poly':
            clf = SVR(kernel='poly', C=1e3, degree=2)
        #    clf = SVR(kernel='linear', C=1e3)
        #    clf = SVR(kernel='linear', C=1e3)


        clf.fit(xTrain, yTrain)
        clf_pred_test = clf.predict(xTest)
        clf_pred_train = clf.predict(xTrain)

        print 'clf_test MSE: ', mean_squared_error(yTest, clf_pred_test)
        print 'clf_train MSE: ', mean_squared_error(yTrain, clf_pred_train)

        print 'clf_test MAE: ', mean_absolute_error(yTest, clf_pred_test)
        print 'clf_train MAE: ', mean_absolute_error(yTrain, clf_pred_train)

        print ' clf test accuracy: ' , sum(1 for x,y in zip(clf_pred_test,yTest ) if x == y) / float(len(yTest))
        print ' clf train accuracy: ' , sum(1 for x,y in zip(clf_pred_train,yTrain ) if x == y) / float(len(yTrain))


    def __init__(self):
        print "-- Created NeuralNetwork Object --"


    def prepare_data(self, train_set, test_set):
        xTrain = train_set.ix[:, :-1].values
        yTrain = train_set.ix[:,-1].values
        xTest = test_set.ix[:, :-1].values
        yTest = test_set.ix[:,-1].values

        print 'xTrain.shape : ' , xTrain.shape
        print 'yTrain.shape: ' , yTrain.shape
        print 'xTest.shape: ' , xTest.shape
        print 'yTest.shape: ' , yTest.shape

        print 'columns: '
        for i in xrange(len(train_set.columns)):
            print 'i: ' , i , ' , column_name: ',  test_set.columns[i]

        yPrevIndex = train_set.columns.tolist().index('label_prev')
        yPrevTrain = train_set.ix[:,yPrevIndex].values
        yPrevTest = test_set.ix[:,yPrevIndex].values
        print ' yPrevIndex : ' , yPrevIndex , 'column_name: ' , train_set.columns[yPrevIndex]

        return xTrain, xTest, yTrain, yTest, yPrevTest, yPrevTrain, yPrevIndex


if __name__ == "__main__":
    # get dataframe for train and test
    # get xTrain and xTest from these dataframes (get_features())
    # get yTrain and yTest from these dataframes (get_labels())
    # build neural network (build_neural_network())


    # arima_df = ARIMA.build_arima_on_labels(table = DB_info.TABLE, county_column_number = COUNTY_COLUMN_NUMBER, train_month = int(0.8 * TOTAL_MONTHS) , order = ( 4, 0 , 2) )
    # arima_df.to_csv('arima_df', sep='\t')


    db_wrapper = DB_wrapper()
    dataframe_train = db_wrapper.retrieve_data(DB_info.FEATURE_TABLE) #get_dataframe(DATABASE, TRAIN_TABLE_NAME)
    dataframe_train = dataframe_train.set_index('cnty_month')


    print 'dataframe_train before: ' , dataframe_train.shape
    # dataframe_train = dataframe_train.dropna(how='any')
    dataframe_train  = dataframe_train.dropna(subset=['label'], how = 'all')
    dataframe_train = dataframe_train[np.isfinite(dataframe_train['label'])]
    print 'dataframe_train after: ' , dataframe_train.shape



    dataframe_train = Util.normalize_each_county(dataframe_train, TOTAL_MONTHS,  NUM_FEATURES)


    Network = NN()
    new_dataframe = Network.merge_with_prev(dataframe_train )

    # new_dataframe = arima_df.join(new_dataframe,  how='inner')
    print 'new_dataframe.columns: ' , new_dataframe.columns
    print 'new_dataframe.shape: ', new_dataframe.shape

    selected_df = new_dataframe[new_dataframe.cnty == 8013]
    selected_df.to_csv(r'data_8013.csv',  sep=',', mode='a', columns= new_dataframe.columns)

    [train_set , test_set] = Network.split_train_test(new_dataframe,  0.8 * TOTAL_MONTHS)
    print 'train shape after split: ' , train_set.shape
    print 'test shape after split : ' , test_set.shape

    cnty_months = ['8013_'+str(i) for i in xrange(45)]

    selected_df = train_set.ix[cnty_months]
    selected_df.to_csv(r'train_8013.csv',  sep=',', mode='a', columns= train_set.columns)
    selected_df = test_set.ix[cnty_months]
    selected_df.to_csv(r'test_8013.csv', sep=',', mode='a', columns= test_set.columns)

    # test_set.drop('label_prev_2', axis=1, inplace=True)
    # train_set.drop('label_prev_2', axis=1, inplace=True)


    # print 'yPrevTest: '
    # print yPrevTest


    # print '99:'
    # print train_set.ix[99,:]
    # print xTrain[99]
    # print yTrain[99]
    #
    # print '100:'
    # print train_set.ix[100,:]
    # print xTrain[100]
    # print yTrain[100]

    # print ' .. train .. '
    # print train_set[train_set.cnty==32005].label
    # print train_set[train_set.cnty==32005].label_prev
    # print train_set[train_set.cnty==32005].label_prev_2

    # print ' .. test ..'
    # print test_set[test_set.cnty==32005].label
    # print test_set[test_set.cnty==32005].label_prev
    # print test_set[test_set.cnty==32005].label_prev_2



    nl_train = train_set[['label_prev_2', 'label_prev' , 'label']]
    nl_test = test_set[['label_prev_2', 'label_prev' , 'label']]
    # Network.compute_baseline( nl_train, nl_test)
    #
    # Network.compute_baseline( train_set, test_set)
    #
    #
    #
    # #linear regression
    # Network.linear_model( nl_train, nl_test , type = 'linear_regression')
    # Network.linear_model( nl_train, nl_test , type = 'ridge_regression')
    #
    # Network.linear_model( train_set, test_set, type = 'linear_regression')
    # Network.linear_model( train_set, test_set, type = 'ridge_regression')

    # Network.linear_classifier('SGDClassifier', train_set, test_set)

    #   Network.linear_classifier('poly', train_set, test_set)
    #   Network.linear_classifier('svm', train_set, test_set)

    Network.neural_net( nl_train, nl_test)
    # Network.neural_net( train_set, test_set)
    #   Network.build_neural_network(xTrain, xTest, yTrain, yTest)
    print "--- Completed ---"

