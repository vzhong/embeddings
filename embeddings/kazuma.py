import tarfile
import numpy as np
from tqdm import tqdm
from embeddings.embedding import Embedding


def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


class KazumaCharEmbedding(Embedding):
    """
    Reference: http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/
    """

    url = 'http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz'
    size = 874474
    d_emb = 100

    def __init__(self, show_progress=True):
        """

        Args:
            show_progress: whether to print progress.

        """

        self.db = self.initialize_db(self.path('kazuma.db'))

        if len(self) < self.size:
            self.clear()
            self.load_word2emb(show_progress=show_progress)

    def emb(self, w, default='zero'):
        assert default == 'zero', 'only zero default is supported for character embeddings'
        chars = ['#BEGIN#'] + list(w) + ['#END#']
        embs = np.zeros(self.d_emb, dtype=np.float32)
        match = {}
        for i in [2, 3, 4]:
            grams = ngrams(chars, i)
            for g in grams:
                g = '{}gram-{}'.format(i, ''.join(g))
                e = self.lookup(g)
                if e is not None:
                    match[g] = np.array(e, np.float32)
        if match:
            embs = sum(match.values()) / len(match)
        return embs.tolist()

    def load_word2emb(self, show_progress=True, batch_size=1000):
        fin_name = self.ensure_file('kazuma.tar.gz', url=self.url)
        seen = set()

        with tarfile.open(fin_name, 'r:gz') as fzip:
            ftxt = fzip.extractfile('charNgram.txt')
            content = ftxt.read()
            ftxt.close()
            lines = content.splitlines()
            if show_progress:
                lines = tqdm(lines)
            batch = []
            for line in lines:
                elems = line.decode().rstrip().split()
                vec = [float(n) for n in elems[-self.d_emb:]]
                word = ' '.join(elems[:-self.d_emb])
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()
            if batch:
                self.insert_batch(batch)


if __name__ == '__main__':
    from time import time
    emb = KazumaCharEmbedding(show_progress=True)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))
