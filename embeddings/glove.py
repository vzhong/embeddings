import random
from collections import namedtuple
from os import path

import zipfile
from tqdm import tqdm
from embeddings.embedding import Embedding


class GloveEmbedding(Embedding):
    """
    Reference: http://nlp.stanford.edu/projects/glove
    """

    GloveSetting = namedtuple('GloveSetting', ['url', 'd_embs', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], 1917494, '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], 2195895, '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], 1193514, '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], 400000, '6B token wikipedia 2014 + gigaword 5'),
    }

    def __init__(self, name='common_crawl_840', d_emb=300, show_progress=True, default='none'):
        """

        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
            default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        assert name in self.settings, '{} is not a valid corpus. Valid options: {}'.format(name, self.settings)
        self.setting = self.settings[name]
        assert d_emb in self.setting.d_embs, '{} is not a valid dimension for {}. Valid options: {}'.format(d_emb, name, self.setting)
        assert default in {'none', 'random', 'zero'}

        self.d_emb = d_emb
        self.name = name
        self.db = self.initialize_db(self.path(path.join('glove', '{}:{}.db'.format(name, d_emb))))
        self.default = default

        if len(self) < self.setting.size:
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
        fin_name = self.ensure_file(path.join('glove', '{}.zip'.format(self.name)), url=self.setting.url)
        seen = set()

        with zipfile.ZipFile(fin_name) as fin:
            fname_zipped = [fzipped.filename for fzipped in fin.filelist if str(self.d_emb) in fzipped.filename][0]
            content = fin.read(fname_zipped)
            lines = content.splitlines()
            if show_progress:
                lines = tqdm(lines, total=self.setting.size)
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
    emb = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))
