import os
import pickle
import numpy as np
import scipy as sp
from scipy.sparse import issparse, spdiags
from scipy.linalg import eig, eigh, svd


def mySVD(X, ReducedDim=0):
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1
    
    if not ReducedDim:
        ReducedDim = 0
    
    mFea, nSmp = X.shape
    
    if nSmp/mFea > 1.0713:
        ddata = X @ X.T
        ddata = np.maximum(ddata, ddata.T)
        
        dimMatrix = ddata.shape[0]
        if (ReducedDim > 0) and (dimMatrix > MAX_MATRIX_SIZE) and (ReducedDim < dimMatrix*EIGVECTOR_RATIO):
            eigvalue, U = eigh(ddata, eigvals=(dimMatrix - ReducedDim, dimMatrix - 1))
            eigvalue = eigvalue[::-1]
            U = U[:, ::-1]
        else:
            if issparse(ddata):
                ddata = ddata.toarray()
                
            eigvalue, U = eigh(ddata)
            eigvalue = eigvalue[::-1]
            U = U[:, ::-1]
        
        eigIdx = np.where(np.abs(eigvalue) / np.max(np.abs(eigvalue)) < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        U = np.delete(U, eigIdx, axis=1)
        
        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):
            eigvalue = eigvalue[:ReducedDim]
            U = U[:, :ReducedDim]
        
        eigvalue_Half = np.sqrt(np.abs(eigvalue))
        S = spdiags(eigvalue_Half, 0, len(eigvalue_Half), len(eigvalue_Half))
        
        eigvalue_MinusHalf = eigvalue_Half ** -1
        V = X.T @ (U * np.tile(eigvalue_MinusHalf, (U.shape[0], 1)))
    else:
        ddata = X.T @ X
        ddata = np.maximum(ddata, ddata.T)
        
        dimMatrix = ddata.shape[0]
        if (ReducedDim > 0) and (dimMatrix > MAX_MATRIX_SIZE) and (ReducedDim < dimMatrix*EIGVECTOR_RATIO):
            eigvalue, V = eigh(ddata, eigvals=(dimMatrix - ReducedDim, dimMatrix - 1))
            eigvalue = eigvalue[::-1]
            V = V[:, ::-1]
        else:
            if issparse(ddata):
                ddata = ddata.toarray()
                
            eigvalue, V = eigh(ddata)
            eigvalue = eigvalue[::-1]
            V = V[:, ::-1]
        
        eigIdx = np.where(np.abs(eigvalue) / np.max(np.abs(eigvalue)) < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        V = np.delete(V, eigIdx, axis=1)
        
        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):
            eigvalue = eigvalue[:ReducedDim]
            V = V[:, :ReducedDim]
        
        eigvalue_Half = eigvalue**0.5
        S =  spdiags(eigvalue_Half,0,len(eigvalue_Half),len(eigvalue_Half))

        eigvalue_MinusHalf = eigvalue_Half**-1

        U = X.dot(V * np.tile(eigvalue_MinusHalf.T,(V.shape[0],1)))

    return (U, S, V)



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
    

    # ====================================
    U, S, V = mySVD(X, ReducedDim=100)
    print("Size of U = ", U.shape)
    print("Size of S = ", S.shape)
    print("Size of V = ", V.shape)