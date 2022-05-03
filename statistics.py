import scipy.io
import numpy as np


def five(filename):
    print(filename)
    data = scipy.io.loadmat(filename)

    n = data['TE'].shape[1]
    m = data['TR'].shape[1]

    print('Number of total documents:', n + m)
    print('Number of training documents:', m)
    print('Number of test documents:', n)

    vs = []
    for v in data['X'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)
    X = np.vstack(vs)
    uni = np.unique(X, axis=0)

    print('Size of dictionary:', len(uni))

    a = [r.shape[1] for r in data['X'][0]]
    ave = np.mean(a)

    print('Unique words in a document: {:.1f}'.format(ave))

    uni = np.unique(data['Y'])

    print('Number of Classes:', len(uni))

    print('Type: five-fold')


def one(filename):
    print(filename)
    data = scipy.io.loadmat(filename)

    n = data['xte'][0].shape[0]
    m = data['xtr'][0].shape[0]

    print('Number of total documents:', n + m)
    print('Number of training documents:', m)
    print('Number of test documents:', n)

    vs = []
    for v in data['xtr'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)
    for v in data['xte'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)
    X = np.vstack(vs)
    uni = np.unique(X, axis=0)

    print('Size of dictionary:', len(uni))

    a = [r.shape[1] for r in data['xte'][0]]
    b = [r.shape[1] for r in data['xtr'][0]]
    ave = np.mean(a + b)

    print('Unique words in a document: {:.1f}'.format(ave))

    uni = np.unique(np.hstack([data['yte'], data['ytr']]))

    print('Number of Classes:', len(uni))

    print('Type: one-fold')


if __name__ == '__main__':
    one('data/ohsumed.mat')
    five('data/classic.mat')
    one('data/reuters.mat')
    five('data/amazon.mat')
    one('data/20news.mat')
