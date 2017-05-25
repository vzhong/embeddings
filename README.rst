Embeddings
==========

.. image:: https://readthedocs.org/projects/embeddings/badge/?version=latest
    :target: http://embeddings.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.org/vzhong/embeddings.svg?branch=master
    :target: https://travis-ci.org/vzhong/embeddings

Embeddings is a python package that provides pretrained word embeddings for natural language processing and machine learning.

Instead of loading a large file to query for embeddings, ``embeddings`` is backed by a database and fast to load and query:

.. code-block:: python

    >>> %timeit GloveEmbedding('common_crawl_840', d_emb=300)
    100 loops, best of 3: 12.7 ms per loop
    
    >>> %timeit GloveEmbedding('common_crawl_840', d_emb=300).emb('canada')
    100 loops, best of 3: 12.9 ms per loop
    
    >>> g = GloveEmbedding('common_crawl_840', d_emb=300)
    
    >>> %timeit -n1 g.emb('canada')
    1 loop, best of 3: 38.2 µs per loop


Installation
------------

.. code-block:: sh

    pip install embeddings  # from pypi
    pip install git+https://github.com/vzhong/embeddings.git  # from github


Usage
-----

Upon first use, the embeddings are first downloaded to disk in the form of a SQLite database.
This may take a long time for large embeddings such as GloVe.
Further usage of the embeddings are directly queried against the database.
Embedding databases are stored in the ``$EMBEDDINGS_ROOT`` directory (defaults to ``~/.embeddings``).


.. code-block:: python

    from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding
    
    g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    f = FastTextEmbedding()
    k = KazumaCharEmbedding()
    for w in ['canada', 'vancouver', 'toronto']:
        print('embedding {}'.format(w))
        print(g.emb(w))
        print(f.emb(w))
        print(k.emb(w))


Contribution
------------

Pull requests welcome!
