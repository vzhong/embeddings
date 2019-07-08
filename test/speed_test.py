#!/usr/bin/env python
import random
import time
from nltk.corpus import brown
import tqdm
import embeddings as E


if __name__ == '__main__':
    random.seed(0)
    n_samples = 10000
    emb = E.GloveEmbedding()
    times = []
    vocab = list(brown.words())
    samples = [random.choice(vocab) for i in range(n_samples)]

    for w in tqdm.tqdm(samples):
        start = time.time()
        emb.emb(w)
        end = time.time()
        times.append(end-start)
    print(sum(times)/len(times))
