import gzip
import random

from collections import namedtuple
from os import path
from tqdm import tqdm

from embeddings.embedding import Embedding


class NumberbatchEmbedding(Embedding):
    """
    Provide different versions of the Conceptnet Numberbatch embeddings published at [1].


    [1]: https://github.com/commonsense/conceptnet-numberbatch
    """

    NumberbatchSetting = namedtuple(
        "NumberbatchSetting",
        ["description", "language", "size", "url", "version"])
    nb_settings = {
        "1908-en": NumberbatchSetting(
            "Numberbatch 19.08 embeddings for English-only tokens.",
            "en",
            516782,
            "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
            "19.08"
        ),
        "1908-ml": NumberbatchSetting(
            "Numberbatch 19.08 embeddings for multiple langauges.",
            "multi",
            9161912,
            "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz",
            "19.08"
        ),
        "1706-en": NumberbatchSetting(
            "Numberbatch 17.06 embeddings for English-only tokens.",
            "en",
            417194,
            "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz",
            "17.06"
        ),
        "1706-ml": NumberbatchSetting(
            "Numberbatch 17.06 embeddings for multiple langauges.",
            "multi",
            1917247,
            "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz",
            "17.06"
        ),
        "1704-en": NumberbatchSetting(
            "Numberbatch 17.04 embeddings for English-only tokens.",
            "en",
            418081,
            "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.04b.txt.gz",
            "17.04"
        ),
        "1704-ml": NumberbatchSetting(
            "Numberbatch 17.04 embeddings for multiple langauges.",
            "multi",
            1918206,
            "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.04.txt.gz",
            "17.04"
        ),
        "1702-en": NumberbatchSetting(
            "Numberbatch 17.02 embeddings for English-only tokens.",
            "en",
            484556,
            "http://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.02.txt.gz",
            "17.02"
        )
    }

    def __init__(self, name="1908-en", show_progress="True", default="none"):
        """
        Arguments:
        name -- Defines the embedding version/langauge combination to be used. Valid values are
                1908-en, 1908-ml, 1706-en, 1706-ml, 1704-en, 1704-ml and 1702-en (en for English,
                ml for multilingual).
        show_progress -- Whether to print a progress bar or not.
        default -- How to embed words that are out-of-vocabulary. Valid values are "none", "zero"
                   and "random".
        """

        # Test if provided parameters are valid
        assert name in self.nb_settings, f"{name} is not a valid name. Valid options are: {self.settings}."
        assert default in {"none", "zero", "random"}

        # Setting default class values
        self.embedding_dimension = 300
        self.name = name
        self.default = default
        self.setting = self.nb_settings[name]
        self.db = self.initialize_db(self.path(path.join("numberbatch", f"{name}.db")))

        # Check if embedding database already exists/is complete, and create/fill it otherwise
        if len(self) < self.setting.size:
            self.clear()
            self.load_word2emb(show_progress=show_progress)

    def load_word2emb(self, show_progress=True, batch_size=1000):
        """Load the word embeddings from a gzipped file and write them to the database.

        Arguments:
        show_progress -- Whether to print a progress bar or not.
        batch_size -- The number of tokens to add to the database at a time.
        """

        # Download embedding file if it does not exist yet
        embedding_file_name = self.ensure_file(
            path.join("numberbatch", f"{self.name}.zip"),
            url=self.setting.url)

        # Open gzipped file and read its content
        with gzip.open(embedding_file_name, "r") as f:
            # Enable the progress bar, if required
            file_content = tqdm(f, total=self.setting.size) if show_progress else f

            # Temporary storage for all unique words
            seen = set()
            # Temporary storage for the currently processed batch
            batch = []
            for line in file_content:
                # Elements in each line are separated by a space, where the first element will
                # always be the string representation of the token and the remaining 300 the vector
                # values.
                line_decoded = line.decode("utf-8")
                all_elements = line_decoded.split(" ")
                token = all_elements[0]
                vector = [float(n) for n in all_elements[1:]]

                if token in seen:
                    continue

                seen.add(token)
                batch.append((token, vector))

                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()

            # Add any remaining tokens to the database
            if batch:
                self.insert_batch(batch)

    def emb(self, word, default=None):
        if default is None:
            default = self.default

        # Produce default vector for out-of-vocabulary tokens
        get_default = {
            "none": lambda: None,
            "zero": lambda: 0.,
            "random": lambda: random.uniform(-0.1, 0.1)
        }[default]

        # Get the actual word vector, if possible
        vector = self.lookup(word)

        return [
            get_default() for i in range(self.embedding_dimension)
        ] if vector is None else vector


if __name__ == '__main__':
    from time import time

    emb = NumberbatchEmbedding('1908-en', show_progress=True)

    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))
