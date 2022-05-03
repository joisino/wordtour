import numpy as np


def load_glove(filepath='glove.6B/glove.6B.300d.txt', n=40000):
    words = []
    embeddings = []
    word_to_embedding = {}
    word_to_id = {}
    with open(filepath) as f:
        for i, r in enumerate(f):
            li = r.split()
            word = li[0]
            embedding = np.array(list(map(float, li[1:])))
            words.append(word)
            embeddings.append(embedding)
            word_to_embedding[word] = embedding
            word_to_id[word] = i
            if len(word_to_id) == n:
                break
    embeddings = np.array(embeddings)
    return words, embeddings, word_to_id, word_to_embedding
