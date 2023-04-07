from PCA1 import PCA1
from ScatterMat import ScatterMat

import os
import pickle
import numpy as np

from tqdm.auto import tqdm

def RSLDA(X=None, label=None, lambda1 = 0.0002,
                              lambda2 = 0.001,
                              dim = 100,
                              mu  = 0.1,
                              rho = 1.01,
                              max_iter = 100):
    print("RUNNING RSLDA!")
    m, n = X.shape
    max_mu = 10**5

    # Initialization
    regu = 10**-5
    Sw, Sb = ScatterMat(X, label)
    options = {}
    options['ReducedDim'] = dim
    P1, _ = PCA1(X.T, options)
    Q = np.ones((m, dim))
    E = np.zeros((m, n))
    Y = np.zeros((m, n))
    v = np.sqrt(np.sum(Q*Q, axis=1) + np.finfo(float).eps)
    D = np.diag(1./v)

    # Main loop
    for iter in tqdm(range(1, max_iter+1), total=max_iter):
        
        # Update P
        if iter == 1:
            P = P1
        else:
            M = X - E + Y/mu
            U1, S1, V1 = np.linalg.svd(M @ X.T @ Q, full_matrices=False)
            P = U1 @ V1
            del M
        
        # Update Q
        M = X - E + Y/mu
        Q1 = 2*(Sw - regu*Sb) + lambda1*D + mu*X @ X.T
        Q2 = mu*X @ M.T @ P
        Q = np.linalg.solve(Q1, Q2)
        v = np.sqrt(np.sum(Q*Q, axis=1) + np.finfo(float).eps)
        D = np.diag(1./v)
        
        # Update E
        eps1 = lambda2/mu
        temp_E = X - P @ Q.T @ X + Y/mu
        E = np.maximum(0, temp_E - eps1) + np.minimum(0, temp_E + eps1)
        
        # Update Y, mu
        Y = Y + mu*(X - P @ Q.T @ X - E)
        mu = min(rho*mu, max_mu)
        leq = X - P @ Q.T @ X - E
        EE = np.sum(np.abs(E), axis=1)
        obj = np.trace(Q.T @ (Sw - regu*Sb) @ Q) + lambda1*np.sum(v) + lambda2*np.sum(EE)
        
        if iter > 2:
            if np.linalg.norm(leq, np.inf) < 10**-7 and abs(obj - obj_prev) < 0.00001:
                print(iter)
                break
        
        obj_prev = obj
    
    return P, Q, E, obj


def sort_power_of_features(Q):
    row_norm = np.linalg.norm(Q, axis=1)
    sorted_power = np.argsort(row_norm)[::-1] # DESC
    return sorted_power


if __name__ == '__main__':
    # DATA_FOLDER = "P:/RESEARCH/DATA/CIFAR-100/CIFARDB/train/"

    features = pickle.load(open("./database/features_handcrafted_Cifar.pkl", 'rb'))
    paths = pickle.load(open("./database/paths_handcrafted_Cifar.pkl", 'rb'))

    # X = np.array(features)
    y = []

    # class_check = 'apple'
    # for i in range(len(paths)):
    #     path = paths[i]
    #     label =  0
    #     if class_check in path:
    #         label = 1
    #     y.append(label)

    X_train = []
    n_pos = 0
    n_neg = 0
    
    class_check = 'apple'
    for i in range(len(paths)):
        if n_pos >= 10 and n_neg >=100:
            break
        else:
            path = paths[i]
            if class_check in path and n_pos < 10:
                label = 1
                n_pos += 1
                y.append(label)
                X_train.append(features[i])
            elif class_check not in path and n_neg < 50:
                label = 0
                n_neg += 1
                y.append(label)
                X_train.append(features[i])
    
    X_train = np.array(X_train)
    X_train = X_train.T
    # X = X.T
    y = np.array(y)
    y = y.reshape(-1, 1)

    print(X_train.shape)
    print(y.shape)
    P,Q,E,obj = RSLDA(X=X_train,label=y)
    print("Shape of P = ", P.shape)
    print("Shape of Q = ", Q.shape)
    print("Shape of E = ", E.shape)

    # pickle.dump(Q, open("./database/Q.pkl", 'wb'))

    # TEST
    # Q = pickle.load(open("./database/Q.pkl", 'rb'))
    # print(Q.shape)
    # print(np.max(Q))
    # print(np.min(Q))
    # a = np.count_nonzero(Q)
    # print(a)
    # sort_power_of_features(Q=Q)
    # print(Q.shape)




