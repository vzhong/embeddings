import random
from collections import namedtuple
from os import path, makedirs

import zipfile
from tqdm import tqdm
from embeddings.embedding import Embedding


class ElmoEmbedding(Embedding):
    """
    Reference: https://allennlp.org/elmo
    """

    settings = {
        'weights': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
        'options': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    }

    def __init__(self):
        from allennlp.modules.elmo import _ElmoCharacterEncoder
        if not path.isdir(self.path('elmo')):
            makedirs(self.path('elmo'))
        self.fweights = self.ensure_file(path.join('elmo', 'weights.hdf5'), url=self.settings['weights'])
        self.foptions = self.ensure_file(path.join('elmo', 'options.json'), url=self.settings['options'])
        self.embeddings = _ElmoCharacterEncoder(self.foptions, self.fweights)

    def emb(self, word, default=None):
        from allennlp.modules.elmo import batch_to_ids
        idx = batch_to_ids([[word]])
        emb = self.embeddings(idx)['token_embedding']
        return emb[0, 1].tolist()


if __name__ == '__main__':
    from time import time
    emb = ElmoEmbedding()
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        print('size {}'.format(len(emb.emb(w))))
        print('took {}s'.format(time() - start))
