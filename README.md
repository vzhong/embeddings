# embeddings

This python package contains utilities to download and make available pretrained word embeddings.

Embeddings are stored in the `$EMBEDDINGS_ROOT` directory in a SQLite 3 database for fast retrieval directory in a SQLite 3 database for minimal load time and fast retrieval.


## Usage

```python
from embeddings import GloVe

emb = GloVe()
```
