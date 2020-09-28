from sklearn.decomposition import PCA
import tools.utils
import pandas as pd
import numpy as np

def demean(X):
    return X - np.mean(X, axis=0)
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):

    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break

        cur_iter += 1

    return w
def first_n_components(n, X, eta=0.01, n_iters = 1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)

        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

    return res

def PCSTransform(data):
    pca = PCA(n_components= 3)
    newdata = pca.fit_transform(data)
    return newdata


    # w = first_n_components(3, data)
    # w = np.array(w)
    # w[:,[0,1,2]] = w[:,[1,0,2]]
    # new_data = np.matmul(data, w)
    # return new_data
