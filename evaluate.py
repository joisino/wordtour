import argparse

import numpy as np
import scipy.io
import scipy.sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from util import load_glove


def blur_bow(X, sigma):
    res = X.copy()
    for i in range(1, 11):
        res += scipy.sparse.hstack([X[:, i:], X[:, :i]]) * np.exp(-(i ** 2) / sigma)
        res += scipy.sparse.hstack([X[:, -i:], X[:, :-i]]) * np.exp(-(i ** 2) / sigma)
    res = normalize(res, axis=1, norm='l1')
    return res


def evaluate_D(y_train, y_test, D, D_train):
    """
    Evaluation using distance metrices

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix of training and test samples
        Shape: (n, m), where n is the number of training documents, m is the number of test documents
        D[i, j] is the distance between training document i and test document j

    D_train : numpy.array
        Distance matrix of training samples
        Shape: (n, n), where n is the number of training documents
        D[i, j] is the distance between training documents i and j

    Returns
    -------
    acc : float
        Accuracy
    """

    parameters = {
        'n_neighbors': [i for i in range(1, 20)]
    }
    clf = GridSearchCV(KNeighborsClassifier(metric='precomputed'), parameters, n_jobs=-1)
    clf.fit(D_train, y_train)
    return float(clf.best_estimator_.score(D, y_test))


def select_k_sigma(y_train, X_train):
    """
    Select the hyperparameter k using validation data

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    X_train : numpy.array
        BOW vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    Returns
    -------
    best_estimator : KNeighborsClassifier
        Chosen model

    best_sigma : int
        Chosen hyperparamter sigma
    """

    best_score = None
    best_sigma = None
    best_estimator = None
    for sigma in [0.01, 0.1, 1.0, 10.0, 100, 1000]:
        X_blur = blur_bow(X_train, sigma)
        D_train = pairwise_distances(X_blur, metric='l1')
        parameters = {
            'n_neighbors': [i for i in range(1, 20)]
        }
        clf = GridSearchCV(KNeighborsClassifier(metric='precomputed'), parameters, n_jobs=-1)
        clf.fit(D_train, y_train)
        if best_score is None or clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_sigma = sigma
            best_estimator = clf.best_estimator_
    return best_estimator, best_sigma


def evaluate_onehot(X_train, y_train, X_test, y_test, blur_flag=True):
    """
    Evaluation using onhot vectors

    Parameters
    ----------
    X_train : numpy.array
        BOW vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    X_test : numpy.array
        BOW vectors of test samples
        Shape: (m, d), where m is the number of test documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    blur_flag: bool
        Blur the BoW or not

    Returns
    -------
    acc : float
        Accuracy
    """

    if blur_flag:
        clf, sigma = select_k_sigma(y_train, X_train)

        X_train = blur_bow(X_train, sigma)
        X_test = blur_bow(X_test, sigma)

        D = pairwise_distances(X_test, X_train, metric='l1')

        return float(clf.score(D, y_test))
    else:
        X_train = normalize(X_train, axis=1, norm='l1')
        X_test = normalize(X_test, axis=1, norm='l1')

        D = pairwise_distances(X_test, X_train, metric='l1')
        D_train = pairwise_distances(X_train, metric='l1')

        return evaluate_D(y_train, y_test, D, D_train)


def five(data, X, y, blur_flag=True):
    """
    Evaluation for five-fold datasets

    Parameters
    ----------
    data : dict
        Dataset dictionary compatible with the original code
        data['TR'] is the indices of the training samples
        data['TE'] is the indices of the test samples
        The indices are 1-indexed

    X : numpy.array
        BOW vectors
        Shape: (n, d), where n is the number of documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y : numpy.array
        Labels
        Shape: (n,), where n is the number of documents
        y[i] is the label of document i

    blur_flag: bool
        Blur the BoW or not

    Returns
    -------
    accs : numpy.array
        accs.shape is (5,)
        Each element represents an accuracy for each fold.
    """

    accs = []
    for i in range(5):
        train = data['TR'][i]
        test = data['TE'][i]
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        accs.append(evaluate_onehot(X_train, y_train, X_test, y_test, blur_flag))
    return np.array(accs)


def fiveD(data, y, D):
    """
    Evaluation for five-fold datasets using a distance matrix

    Parameters
    ----------
    data : dict
        Dataset dictionary compatible with the original code
        data['TR'] is the indices of the training samples
        data['TE'] is the indices of the test samples
        The indices are 1-indexed

    y : numpy.array
        Labels
        Shape: (n,), where n is the number of documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix
        Shape: (n, n), where n is the number of documents
        D[i, j] is the distance between documents i and j

    Returns
    -------
    accs : numpy.array
        accs.shape is (5,)
        Each element represents an accuracy for each fold.
    """

    accs = []
    for i in range(5):
        train = data['TR'][i]
        test = data['TE'][i]
        y_train = y[train]
        y_test = y[test]
        accs.append(evaluate_D(y_train, y_test, D[test][:, train], D[train][:, train]))
    return np.array(accs)


def load_order(filepath, word_to_id):
    order = []
    with open(filepath) as f:
        for r in f:
            order.append(word_to_id[r.strip()])
    return order


def evaluate_five(filename, wmd_flag):
    print(filename)
    print('-' * len(filename))
    data = scipy.io.loadmat('data/{}'.format(filename))
    X = data['bow']
    y = data['Y'][0]

    res = {}
    res['BoW'] = (1 - five(data, X, y, blur_flag=False)) * 100
    print('BOW {:.1f} {:.1f}'.format(res['BoW'].mean(), res['BoW'].std()))

    _, _, word_to_id, _ = load_glove()
    for order_filename in ['wordtour.txt', 'order_randproj.txt', 'order_pca1.txt', 'order_pca4.txt']:
        order = load_order(order_filename, word_to_id)
        res[order_filename] = (1 - five(data, X[:, order], y)) * 100
        print('{} {:.1f} {:.1f}'.format(order_filename, res[order_filename].mean(), res[order_filename].std()))

    if wmd_flag:
        D = np.load('distance/{}.npy'.format(filename))

        res['WMD'] = (1 - fiveD(data, y, D)) * 100
        print('WMD\t{:.1f} ± {:.1f}'.format(res['WMD'].mean(), res['WMD'].std()))

    print()

    return res


def evaluate_one(filename, wmd_flag):
    print(filename)
    print('-' * len(filename))
    data = scipy.io.loadmat('data/{}'.format(filename))
    X_train = data['bow_tr']
    y_train = data['ytr'][0]
    X_test = data['bow_te']
    y_test = data['yte'][0]

    res = {}
    res['BoW'] = (1 - evaluate_onehot(X_train, y_train, X_test, y_test, blur_flag=False)) * 100
    print('BOW\t{:.1f}'.format(res['BoW']))

    _, _, word_to_id, _ = load_glove()
    for order_filename in ['wordtour.txt', 'order_randproj.txt', 'order_pca1.txt', 'order_pca4.txt']:
        order = load_order(order_filename, word_to_id)
        res[order_filename] = (1 - evaluate_onehot(X_train[:, order], y_train, X_test[:, order], y_test)) * 100
        print('{} {:.1f}'.format(order_filename, res[order_filename]))

    if wmd_flag:
        D = np.load('distance/{}.npy'.format(filename))
        D_train = np.load('distance/{}-train.npy'.format(filename))

        res['WMD'] = (1 - evaluate_D(y_train, y_test, D, D_train)) * 100
        print('WMD\t{:.1f}'.format(res['WMD']))

    print()

    return res


def output_summary(result):
    methods = [
        ('BoW', 'BoW'),
        ('wordtour.txt', 'WordTour'),
        ('order_randproj.txt', 'RandProj'),
        ('order_pca1.txt', 'PCA1'),
        ('order_pca4.txt', 'PCA4'),
        ('WMD', 'WMD'),
    ]
    for method, method_name in methods:
        li = []
        for result in results:
            if type(result[method]) == float:
                text = '{:.1f}'.format(result[method])
            elif type(result[method]) == np.ndarray:
                text = '{:.1f} $\\pm$ {:.1f}'.format(result[method].mean(), result[method].std())
            li.append(text)
        print(method_name + ' & ' + ' & '.join(li) + ' \\\\')

    base = 'wordtour.txt'
    li = []
    for result in results:
        if type(result[base]) == float:
            base_val = result[base]
            bow_val = result['BoW']
            wmd_val = result['WMD']
        elif type(result[base]) == np.ndarray:
            base_val = result[base].mean()
            bow_val = result['BoW'].mean()
            wmd_val = result['WMD'].mean()
        improve_ratio = (1 - (base_val - wmd_val) / (bow_val - wmd_val)) * 100
        li.append('{:.1f}'.format(improve_ratio))
    print('rel.' + ' & ' + ' & '.join(li) + ' \\\\')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wmd', action='store_true')
    args = parser.parse_args()
    datasets = [
        ('ohsumed.mat', 'one'),
        ('classic.mat', 'five'),
        ('reuters.mat', 'one'),
        ('amazon.mat', 'five'),
        ('20news.mat', 'one'),
    ]
    results = []
    for dataset, fold in datasets:
        if fold == 'five':
            result = evaluate_five(dataset, args.wmd)
        elif fold == 'one':
            result = evaluate_one(dataset, args.wmd)
        results.append(result)
    if args.wmd:
        output_summary(results)
