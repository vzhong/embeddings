#!/usr/bin/env python
import random
import time
from nltk.corpus import brown
import tqdm
import embeddings as E


if __name__ == '__main__':
    random.seed(0)
    n_samples = 10000
    k = E.KazumaCharEmbedding()

    for w in ['canada', 'vancouver', 'toronto']:
        print(k.emb(w))
        
