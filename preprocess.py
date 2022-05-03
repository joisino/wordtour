import os
import codecs
import scipy.io
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from util import load_glove


_, _, word_to_id, word_to_embedding = load_glove()

stopwords = []
with open('WMD_text_datasets/stop_words_115.txt') as f:
    for r in f:
        stopwords.append(r.strip())


def load(filename, label_ma=None, remove_stopwords=True):
    row = []
    col = []
    data = []
    labels = []
    if label_ma is None:
        label_ma = {}
    X = []
    with codecs.open(filename, 'r', 'utf-8', 'ignore') as f:
        for r in f:
            vecs = []
            for w in r.split()[1:]:
                if not (remove_stopwords and w in stopwords):
                    if w in word_to_id:
                        vecs.append(word_to_embedding[w])
                        w = word_to_id[w]
                        data.append(1)
                        col.append(w)
                        row.append(len(labels))
            if len(vecs) == 0:
                continue
            label = r.split()[0]
            if label not in label_ma:
                label_ma[label] = len(label_ma)
            label = label_ma[label]
            labels.append(label)
            X.append(np.array(vecs).T)
    X = np.array(X + [[]], dtype=object)[:-1].reshape(1, -1)
    y = np.array(labels)
    return X, y, data, row, col, label_ma


def five(rawfile, tofile, remove_stopwords=True):
    print(rawfile)
    X, y, data, row, col, _ = load(rawfile, remove_stopwords=remove_stopwords)
    bow = csr_matrix(coo_matrix((data, (row, col)), shape=(len(y), len(word_to_id))))

    n = len(y)
    D = pairwise_distances(bow)
    D[np.arange(n), np.arange(n)] = 1
    dup = (D == 0)
    mask = np.ones(n, dtype=int)
    np.random.seed(0)
    for i in np.random.permutation(n):
        if (dup[i] * mask).sum() > 0:
            mask[i] = 0
    ind = np.arange(n)[mask == 1]
    bow = bow[ind]
    X = X[:, ind]
    y = y[ind]

    train = []
    test = []
    for i in range(5):
        train_i, test_i = train_test_split(np.arange(len(y)), test_size=0.3, random_state=i)
        train.append(train_i)
        test.append(test_i)
    train = np.vstack(train)
    test = np.vstack(test)
    dic = {
        'X': X,
        'Y': y,
        'TR': train,
        'TE': test,
        'bow': bow,
    }
    scipy.io.savemat(tofile, dic)


def one(rawfile_train, rawfile_test, tofile, remove_stopwords=True):
    print(rawfile_train, rawfile_test)
    xtr, ytr, data_tr, row_tr, col_tr, label_ma = load(rawfile_train, remove_stopwords=remove_stopwords)
    xte, yte, data_te, row_te, col_te, label_ma = load(rawfile_test, label_ma, remove_stopwords=remove_stopwords)
    row_te = [row + len(ytr) for row in row_te]
    bow = csr_matrix(coo_matrix((data_tr + data_te, (row_tr + row_te, col_tr + col_te)), shape=(len(ytr) + len(yte), len(word_to_id))))

    n = len(ytr) + len(yte)
    D = pairwise_distances(bow)
    D[np.arange(n), np.arange(n)] = 1
    dup = (D == 0)
    mask = np.ones(n, dtype=int)
    np.random.seed(0)
    for i in np.random.permutation(n):
        if (dup[i] * mask).sum() > 0:
            mask[i] = 0
    ind = np.arange(n)[mask == 1]
    bow = bow[ind]
    xte = xte[:, ind[ind >= len(ytr)] - len(ytr)]
    xtr = xtr[:, ind[ind < len(ytr)]]
    yte = yte[ind[ind >= len(ytr)] - len(ytr)]
    ytr = ytr[ind[ind < len(ytr)]]

    bow_tr = bow[:len(ytr)]
    bow_te = bow[len(ytr):]
    dic = {
        'xtr': xtr,
        'xte': xte,
        'ytr': ytr,
        'yte': yte,
        'bow_tr': bow_tr,
        'bow_te': bow_te
    }
    scipy.io.savemat(tofile, dic)


if not os.path.exists('data'):
    os.mkdir('data')

one('WMD_text_datasets/train_ohsumed_by_line.txt', 'WMD_text_datasets/test_ohsumed_by_line.txt', 'data/ohsumed.mat')
five('WMD_text_datasets/all_classic.txt', 'data/classic.mat')
one('WMD_text_datasets/r8-train-all-terms.txt', 'WMD_text_datasets/r8-test-all-terms.txt', 'data/reuters.mat')
five('WMD_text_datasets/all_amazon_by_line.txt', 'data/amazon.mat')
one('WMD_text_datasets/20ng-train-all-terms.txt', 'WMD_text_datasets/20ng-test-all-terms.txt', 'data/20news.mat')
