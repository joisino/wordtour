import time
import numpy as np
import scipy.io
import scipy.sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import ot

from util import load_glove
from evaluate import blur_bow, load_order


def compute_distance(X, blur_flag):
    if blur_flag:
        X = blur_bow(X, 10.0)
    else:
        X = normalize(X, axis=1, norm='l1')

    D = pairwise_distances(X, X, metric='l1')
    return D


def evaluate(bow, X):
    n = bow.shape[0]

    start = time.time()
    compute_distance(bow, False)
    end = time.time()
    par_element_bow = (end - start) / (n * n)
    print('BoW {:.0f} ns'.format(par_element_bow * 1e9))

    _, _, word_to_id, _ = load_glove()
    order = load_order('wordtour.txt', word_to_id)
    start = time.time()
    compute_distance(bow[:, order], True)
    end = time.time()
    par_element_wordtour = (end - start) / (n * n)
    print('WordTour {:.0f} ns'.format(par_element_wordtour * 1e9))

    N = 1000
    np.random.seed(0)
    start = time.time()
    for _ in range(N):
        i, j = np.random.randint(n, size=2)
        D = pairwise_distances(X[0, i].T, X[0, j].T)
        a = np.ones(D.shape[0]) / D.shape[0]
        b = np.ones(D.shape[1]) / D.shape[1]
        T = ot.emd(a, b, D)
        _ = (T * D).sum()
    end = time.time()
    par_element_wmd = (end - start) / N
    print('WMD {:.2f} ms'.format(par_element_wmd * 1e3))
    print()

    return par_element_bow, par_element_wordtour, par_element_wmd


def evaluate_five(filename):
    print(filename)
    print('-' * len(filename))
    data = scipy.io.loadmat('data/{}'.format(filename))
    return evaluate(data['bow'], data['X'])


def evaluate_one(filename):
    print(filename)
    print('-' * len(filename))
    data = scipy.io.loadmat('data/{}'.format(filename))
    return evaluate(data['bow_tr'], data['xtr'])


def output_summary(results):
    print('BoW', end=' & ')
    for i, result in enumerate(results):
        end = ' \\\\\n' if i == len(results) - 1 else ' & '
        print('{:.0f} \\textbf{{ns}}'.format(result[0] * 1e9), end=end)

    print('WordTour', end=' & ')
    for i, result in enumerate(results):
        end = ' \\\\\n' if i == len(results) - 1 else ' & '
        print('{:.0f} \\textbf{{ns}}'.format(result[1] * 1e9), end=end)

    print('WMD', end=' & ')
    for i, result in enumerate(results):
        end = ' \\\\\n' if i == len(results) - 1 else ' & '
        print('{:.2f} \\textbf{{ms}}'.format(result[2] * 1e3), end=end)


if __name__ == '__main__':
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
            result = evaluate_five(dataset)
        elif fold == 'one':
            result = evaluate_one(dataset)
        results.append(result)
    output_summary(results)
