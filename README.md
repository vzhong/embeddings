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


## Usage

```python
from embeddings import GloveEmbedding

emb = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
for w in ['canada', 'vancouver', 'toronto']:
    print('embedding {}'.format(w))
    print(emb.emb(w))
```

## Contribution

Pull requests welcome!
