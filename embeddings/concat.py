from embeddings.embedding import Embedding


class ConcatEmbedding(Embedding):
    """
    A concatenation of multiple embeddings
    """

    def __init__(self, embeddings, default='none'):
        """

        Args:
            embeddings: embeddings to concatenate.
            default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        for e in embeddings:
            assert isinstance(e, Embedding), '{} is not an Embedding object'.format(e)
        assert default in {'none', 'random', 'zero'}

        self.embeddings = embeddings
        self.default = default

    def emb(self, word, default=None):
        if default is None:
            default = self.default
        emb = []
        for e in self.embeddings:
            emb += e.emb(word, default=default)
        return emb
