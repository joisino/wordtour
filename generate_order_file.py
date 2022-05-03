import numpy as np
from sklearn.decomposition import PCA
from util import load_glove


def load_ordering():
    with open('LKH-3.0.6/wordtour.out') as f:
        lkh = f.readlines()
    n = len(lkh[6:-2])

    words, emb, _, _ = load_glove(n=n)

    np.random.seed(0)
    rand_proj = np.argsort(emb @ np.random.randn(300))
    rand_proj = [words[i] for i in rand_proj]

    pca = PCA(n_components=4, random_state=0)
    pca = pca.fit_transform(emb)
    pca1 = np.argsort(pca[:, 0])
    pca1 = [words[i] for i in pca1]
    pca4 = np.argsort(pca[:, 3])
    pca4 = [words[i] for i in pca4]

    lkh = [words[i - 1] for i in map(int, lkh[6:-2])]

    return words, lkh, rand_proj, pca1, pca4, emb


def save_ordering(lkh, rand_proj, pca1, pca4):
    output_list = [
        ('wordtour.txt', lkh),
        ('order_randproj.txt', rand_proj),
        ('order_pca1.txt', pca1),
        ('order_pca4.txt', pca4)
    ]

    for filename, ordering in output_list:
        with open(filename, 'w') as f:
            for w in ordering:
                print(w, file=f)


def save_amt_csv(words, lkh, rand_proj, pca1, pca4):
    with open('amt.csv', 'w') as f:
        print('text1,text2,gt1,gt2', file=f)
        for i in range(300):
            np.random.seed(i)
            w = lkh[np.random.randint(len(words))]
            lkh_index = lkh.index(w)
            rand_proj_index = rand_proj.index(w)
            pca1_index = pca1.index(w)
            pca4_index = pca4.index(w)
            lkh_seg = ', '.join(lkh[lkh_index - 5:lkh_index + 6])
            rand_proj_seg = ', '.join(rand_proj[rand_proj_index - 5:rand_proj_index + 6])
            pca1_seg = ', '.join(pca1[pca1_index - 5:pca1_index + 6])
            pca4_seg = ', '.join(pca1[pca4_index - 5:pca4_index + 6])
            if i < 100:
                base_seg = rand_proj_seg
                base_name = 'rand'
            elif i < 200:
                base_seg = pca1_seg
                base_name = 'pca1'
            else:
                base_seg = pca4_seg
                base_name = 'pca4'
            lkh_seg = lkh_seg.replace('"', '""')
            base_seg = base_seg.replace('"', '""')
            if np.random.randint(2) == 0:
                print('"' + lkh_seg + '",' + '"' + base_seg + '",' + 'lkh,{}'.format(base_name), file=f)
            else:
                print('"' + base_seg + '",' + '"' + lkh_seg + '",' + '{},lkh'.format(base_name), file=f)


def save_amt_word_csv(words, lkh, rand_proj, pca1, pca4):
    n = len(words)
    with open('amt_word.csv', 'w') as f:
        print('center,word1,word2,gt1,gt2', file=f)
        for i in range(300):
            np.random.seed(i)
            w = lkh[np.random.randint(len(words))]
            lkh_index = lkh.index(w)
            rand_proj_index = rand_proj.index(w)
            pca1_index = pca1.index(w)
            pca4_index = pca4.index(w)
            lkh_word = lkh[(lkh_index + 1) % n]
            if i < 100:
                base_word = rand_proj[(rand_proj_index + 1) % n]
                base_name = 'rand'
            elif i < 200:
                base_word = pca1[(pca1_index + 1) % n]
                base_name = 'pca1'
            else:
                base_word = pca4[(pca4_index + 1) % n]
                base_name = 'pca4'
            lkh_word = lkh_word.replace('"', '""')
            base_word = base_word.replace('"', '""')
            if np.random.randint(2) == 0:
                print('"' + w + '","' + lkh_word + '","' + base_word + '",' + 'lkh,{}'.format(base_name), file=f)
            else:
                print('"' + w + '","' + base_word + '","' + lkh_word + '",' + '{},lkh'.format(base_name), file=f)


def save_amt_word_glove_csv(words, lkh, emb):
    n = len(words)
    with open('amt_word_glove.csv', 'w') as f:
        print('center,word1,word2,gt1,gt2', file=f)
        for i in range(300, 400):
            np.random.seed(i)
            w = lkh[np.random.randint(len(words))]
            lkh_index = lkh.index(w)
            emb_index = words.index(w)
            lkh_word = lkh[(lkh_index + 1) % n]
            base_word = words[np.argsort(np.linalg.norm(emb - emb[emb_index], axis=1))[1]]
            if lkh_word == base_word:
                continue
            lkh_word = lkh_word.replace('"', '""')
            base_word = base_word.replace('"', '""')
            if np.random.randint(2) == 0:
                print('"' + w + '","' + lkh_word + '","' + base_word + '",' + 'lkh,glove', file=f)
            else:
                print('"' + w + '","' + base_word + '","' + lkh_word + '",' + 'glove,lkh', file=f)


def main():
    words, lkh, rand_proj, pca1, pca4, emb = load_ordering()
    save_ordering(lkh, rand_proj, pca1, pca4)
    save_amt_csv(words, lkh, rand_proj, pca1, pca4)
    save_amt_word_csv(words, lkh, rand_proj, pca1, pca4)
    save_amt_word_glove_csv(words, lkh, emb)


if __name__ == '__main__':
    main()
