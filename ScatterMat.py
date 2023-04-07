import os
import numpy as np
import pickle


def ScatterMat(X, y):
    dim, _ = X.shape
    nclass = int(np.max(y)) + 1
    
    mean_X = np.mean(X, axis=1)
    Sw = np.zeros((dim,dim))
    Sb = np.zeros((dim,dim))
    
    for i in range(nclass):
        inx_i = np.where(y==i)[0]
        X_i = X[:,inx_i]
        
        mean_Xi = np.mean(X_i, axis=1)
        Sw += np.cov(X_i, rowvar=True, bias=True)
        Sb += len(inx_i)*(mean_Xi-mean_X)[:,None]*(mean_Xi-mean_X)[None,:]
        
    return Sw, Sb


if __name__ == '__main__':
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
    X = X.T
    y = y.reshape(-1, 1)
    print(X.shape)
    print(y.shape)
    Sw, Sb = ScatterMat(X, y)
    print(Sw.shape, Sb.shape)
    print(np.max(Sw), np.max(Sb))
