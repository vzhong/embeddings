FROM continuumio/miniconda3

ENV EMBEDDINGS_ROOT /opt/embeddings
RUN mkdir -p $EMBEDDINGS_ROOT/code
WORKDIR $EMBEDDINGS_ROOT/code
COPY . .

RUN pip install .[test]
RUN nosetests test
RUN python -c 'from embeddings import GloveEmbedding; GloveEmbedding()'
RUN python -c 'from embeddings import KazumaCharEmbedding; KazumaCharEmbedding()'

VOLUME $EMBEDDINGS_ROOT
