import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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


def normalize_min_max(dataset, train_size, axis = 0):
    minimum = np.min(dataset[: train_size], axis = axis)
    maximum = np.max(dataset[: train_size], axis = axis)

    dataset = ((dataset - minimum)* 100.0)/ (maximum-minimum)
    # dataset = (dataset - mean)/standard_deviation
    return dataset

def normalize_min_max_splitted(train, test):
    minimum = np.min(train, axis = 0)
    maximum = np.max(train, axis = 0)
    print 'minimum'
    # print minimum[1:20]
    print 'maximum'
    # print maximum[1:20]
    print 'min: ' , minimum.shape

    train = ((train - minimum)* 100.0)/ (maximum-minimum)
    test = ((test - minimum)* 100.0)/ (maximum-minimum)
    # dataset = (dataset - mean)/standard_deviation
    return [train, test]

# This method normalizes the data using the mean and
# standard deviation, obtained using the train data
# only.
# Note: This happens column wise, because the data has
# been transposed.
# Data is in the form:
# County ......
# Month1 ......
# MonthN ......
def normalize_mean_variance(dataset, train_size):
    mean = np.mean(dataset[: train_size], axis = 0)

    standard_deviation = np.std(dataset[: train_size], axis = 0)

    dataset = (dataset - mean) * 10.0/standard_deviation
    return dataset

def normalize_mean_variance_splitted(train, test):
    mean = np.mean(train, axis = 0)

    standard_deviation = np.std(train, axis = 0)

    train = (train - mean) * 10.0/standard_deviation
    test = (test - mean) * 10.0/standard_deviation
    return [train, test]


def normalize_each_county(df,num_of_months , num_of_features ):
    counties  = df.cnty.unique().tolist()
    # num_of_months = 45
    # num_of_features = 78
    months = [ i for i in xrange(num_of_months)]
    features = ['feat_'+str(i) for i in xrange(num_of_features)]
    features.append('label')

    not_complete = []
    for feat in features:
        #print 'Feat: ', feat
        for county in counties:
            values = []
            br = 0
            for month in months:
                idx = str(county) +'_'+str(month)
                #print  ' cnty_month: ' , county , ' , ', month , ' ,',  idx, ' , ', feat
                if idx in df.index:
                    val = df.get_value(idx, feat)
                    if val is None:
                        print county , ' , ', month
                else:
                    if county not in not_complete:
                        not_complete.append(county)
                    val = None
                    br = 1
                    break
                values.append(val)

            if br == 0:
                # new_values = normalize_mean_variance(values, int(num_of_months * 0.8))
                new_values = normalize_min_max(values, int(num_of_months * 0.8))
                #print 'max, min: ' , np.max(new_values), ' , ', np.min(new_values)
                for j in xrange(num_of_months):
                    idx = str(county) +'_'+str(j)
                    df.set_value(idx, feat , new_values[j])
        '''
	    if county == '33007':
                print 'county: ' , county
                print 'values:\n\n'
		print values
		print 'new_values:\n\n'
		print new_values
    	'''

    print 'df.shape: ' , df.shape
    print 'note_complete: ' , not_complete

    df = df[-df['cnty'].isin(not_complete)]
    print 'df.shape: ' , df.shape
    print 'len(not_complete) : ', len(not_complete)
    return df