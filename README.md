# embeddings

This python package contains utilities to download and make available pretrained word embeddings.

Embeddings are stored in the `$EMBEDDINGS_ROOT` directory (defaults to `~/.embeddings`) in a SQLite 3 database for minimal load time and fast retrieval.

Instead of loading a large file to query for embeddings, `embeddings` is fast:

```python
In [1]: %timeit GloveEmbedding('common_crawl_840', d_emb=300)
100 loops, best of 3: 12.7 ms per loop

In [2]: %timeit GloveEmbedding('common_crawl_840', d_emb=300).emb('canada')
100 loops, best of 3: 12.9 ms per loop

In [3]: g = GloveEmbedding('common_crawl_840', d_emb=300)

In [4]: %timeit -n1 g.emb('canada')
1 loop, best of 3: 38.2 µs per loop
```

## Installation

```bash
pip install embeddings  # from pypi
pip install git+https://github.com/vzhong/embeddings.git  # from github
```


## Usage

Note that on first usage, the embeddings will be downloaded. This may take a long time for large embeddings such as GloVe.

```python
from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
f = FastTextEmbedding()
k = KazumaCharEmbedding()
for w in ['canada', 'vancouver', 'toronto']:
    print('embedding {}'.format(w))
    print(g.emb(w))
    print(f.emb(w))
    print(k.emb(w))
```

## Contribution

Pull requests welcome!
