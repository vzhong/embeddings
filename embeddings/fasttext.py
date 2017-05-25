import random
from collections import namedtuple
from os import path

import zipfile
from tqdm import tqdm
from embeddings.embedding import Embedding


class FastTextEmbedding(Embedding):
    """
    Reference: https://arxiv.org/abs/1607.04606
    """

    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.zip'
    sizes = {
        'en': 1,
    }
    d_emb = 300

    def __init__(self, lang='en', show_progress=True, default='none'):
        """

        Args:
            lang (en): what language to use.
            show_progress (bool): whether to print progress.
            default (str): how to embed words that are out of vocabulary.

        Note:
            Default can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        assert default in {'none', 'random', 'zero'}

        self.lang = lang
        self.db = self.initialize_db(self.path(path.join('fasttext', '{}.db'.format(lang))))
        self.default = default

        if len(self) < self.sizes[self.lang]:
            self.clear()
            self.load_word2emb(show_progress=show_progress)

    def emb(self, word, default=None):
        if default is None:
            default = self.default
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        g = self.lookup(word)
        return [get_default() for i in range(self.d_emb)] if g is None else g

    def load_word2emb(self, show_progress=True, batch_size=1000):
        fin_name = self.ensure_file(path.join('fasttext', '{}.zip'.format(self.lang)), url=self.url.format(self.lang))
        seen = set()

        with zipfile.ZipFile(fin_name) as fin:
            content = fin.read('wiki.{}.vec'.format(self.lang))
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
    emb = FastTextEmbedding('en', show_progress=True)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))
