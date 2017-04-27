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


def normalize_min_max(dataset, train_size):
    minimum = np.min(dataset[1: train_size], axis = 0)
    maximum = np.max(dataset[1: train_size], axis = 0)
    print 'minimum'
    # print minimum[1:20]
    print 'maximum'
    # print maximum[1:20]
    print 'min: ' , minimum.shape
    # standard_deviation = np.std(dataset[1: train_size], axis = 0)
    # print 'standard_deviation: ' , standard_deviation.shape
    # print 'dataset: ' , dataset.shape
    dataset = ((dataset - minimum)* 100.0)/ (maximum-minimum)
    # dataset = (dataset - mean)/standard_deviation
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
def normalize_mean_variance(dataset, train_size):
    mean = np.mean(dataset[1: train_size], axis = 0)
    print 'mean: ' , mean.shape
    standard_deviation = np.std(dataset[1: train_size], axis = 0)
    print 'standard_deviation: ' , standard_deviation.shape
    print 'dataset: ' , dataset.shape
    dataset = (dataset - mean) * 10.0/standard_deviation
    return dataset