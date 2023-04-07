import os
import pickle
from mySVD import mySVD
from scipy.sparse import issparse

import numpy as np


def PCA1(data, options=None):
    # Principal Component Analysis

    # Input:
    # data - Data matrix. Each row vector of data is a data point.
    # options.ReducedDim - The dimensionality of the reduced subspace. If 0, all the dimensions will be kept. Default is 0.
    # Output:
    # eigvector - Each column is an embedding function, for a new data point (row vector) x, y = x*eigvector will be the embedding result of x.
    # eigvalue - The sorted eigvalue of PCA eigen-problem.

    if options is None:
        options = {}

    ReducedDim = 0
    if 'ReducedDim' in options:
        ReducedDim = options['ReducedDim']

    nSmp, nFea = data.shape
    if (ReducedDim > nFea) or (ReducedDim <=0):
        ReducedDim = nFea

    if issparse(data):
        data = data.toarray()
    sampleMean = np.mean(data, axis=0)
    data = (data - np.tile(sampleMean, (nSmp, 1)))

    eigvector, eigvalue, _ = mySVD(data.T, ReducedDim)
    eigvalue = np.square(eigvalue)

    if 'PCARatio' in options:
        sumEig = np.sum(eigvalue)
        sumEig *= options['PCARatio']
        sumNow = 0
        for idx in range(len(eigvalue)):
            sumNow += eigvalue[idx]
            if sumNow >= sumEig:
                break
        eigvector = eigvector[:, :idx]

    return eigvector, eigvalue


if __name__ == '__main__':
    dim = 100
    options = {
        "ReducedDim": dim
    }

    DATA_FOLDER = "P:/RESEARCH/DATA/CIFAR-100/CIFARDB/train/"
    features = pickle.load(open("./database/features_YCrCb_CifarDataset.pkl", 'rb'))
    paths = pickle.load(open("./database/paths_YCrCb_CifarDataset.pkl", 'rb'))
    features = features
    paths = paths
    classes = os.listdir(DATA_FOLDER)
    X = np.array(features)
    y = []
    class_check = 'apple'
    for path in paths:
        label = 0
        if class_check in path:
            label = 1
        y.append(label)
    y = np.array(y)
    y = y.reshape(-1, 1)

    eigvector, eigvalue = PCA1(X, options)
    print("Shape of eigvector = ", eigvector.shape)
    print("Shape of eigvalue = ", eigvalue.shape)

