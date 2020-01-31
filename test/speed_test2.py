#!/usr/bin/env python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import random
import time
from nltk.corpus import brown
import tqdm
import embeddings as E


if __name__ == '__main__':
    random.seed(0)
    n_samples = 10000
    k1 = E.KazumaCharEmbedding(check_same_thread=True)
    k2 = E.KazumaCharEmbedding(check_same_thread=False)

    g1 = E.GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True, check_same_thread=True)
    g2 = E.GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True, check_same_thread=False)

    for w in ['canada', 'vancouver', 'toronto']:
        assert(k1.emb(w) == k2.emb(w))
        assert(g1.emb(w) == g2.emb(w))

        
