import sqlite3
from os import path, makedirs, environ
import requests
import logging
from array import array


class Embedding:

    @staticmethod
    def path(p):
        """

        Args:
            p (str): relative path.

        Returns:
            str: absolute path to the file, located in the ``$EMBEDDINGS_ROOT`` directory.

        """
        root = environ.get('EMBEDDINGS_ROOT', path.join(environ['HOME'], '.embeddings'))
        return path.join(path.abspath(root), p)

    @staticmethod
    def download_file(url, local_filename):
        """
        Downloads a file from an url to a local file.

        Args:
            url (str): url to download from.
            local_filename (str): local file to download to.

        Returns:
            str: file name of the downloaded file.

        """
        r = requests.get(url, stream=True)
        if path.dirname(local_filename) and not path.isdir(path.dirname(local_filename)):
            raise Exception(local_filename)
            makedirs(path.dirname(local_filename))
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return local_filename

    @staticmethod
    def ensure_file(name, url=None, force=False, logger=logging.getLogger(), postprocess=None):
        """
        Ensures that the file requested exists in the cache, downloading it if it does not exist.

        Args:
            name (str): name of the file.
            url (str): url to download the file from, if it doesn't exist.
            force (bool): whether to force the download, regardless of the existence of the file.
            logger (logging.Logger): logger to log results.
            postprocess (function): a function that, if given, will be applied after the file is downloaded. The function has the signature ``f(fname)``

        Returns:
            str: file name of the downloaded file.

        """
        fname = Embedding.path(name)
        if not path.isfile(fname) or force:
            if url:
                logger.critical('Downloading from {} to {}'.format(url, fname))
                Embedding.download_file(url, fname)
                if postprocess:
                    postprocess(fname)
            else:
                raise Exception('{} does not exist!'.format(fname))
        return fname

    @staticmethod
    def initialize_db(fname):
        """

        Args:
            fname (str): location of the database.

        Returns:
            db (sqlite3.Connection): a SQLite3 database with an embeddings table.

        """
        if path.dirname(fname) and not path.isdir(path.dirname(fname)):
            makedirs(path.dirname(fname))
        db = sqlite3.connect(fname)
        c = db.cursor()
        c.execute('create table if not exists embeddings(word text primary key, emb blob)')
        db.commit()
        return db

    def __len__(self):
        """

        Returns:
            count (int): number of embeddings in the database.

        """
        c = self.db.cursor()
        q = c.execute('select count(*) from embeddings')
        self.db.commit()
        return q.fetchone()[0]

    def insert_batch(self, batch):
        """

        Args:
            batch (list): a list of embeddings to insert, each of which is a tuple ``(word, embeddings)``.

        Example:

        .. code-block:: python

            e = Embedding()
            e.db = e.initialize_db(self.e.path('mydb.db'))
            e.insert_batch([
                ('hello', [1, 2, 3]),
                ('world', [2, 3, 4]),
                ('!', [3, 4, 5]),
            ])
        """
        c = self.db.cursor()
        binarized = [(word, array('f', emb).tobytes()) for word, emb in batch]
        try:
            c.executemany("insert into embeddings values (?, ?)", binarized)
            self.db.commit()
        except Exception as e:
            print('insert failed\n{}'.format([w for w, e in batch]))
            raise e

    def __contains__(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            whether an embedding for ``w`` exists.

        """
        return self.lookup(w) is not None

    def clear(self):
        """

        Deletes all embeddings from the database.

        """
        c = self.db.cursor()
        c.execute('delete from embeddings')
        self.db.commit()

    def lookup(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.

        """
        c = self.db.cursor()
        q = c.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
        return array('f', q[0]).tolist() if q else None
