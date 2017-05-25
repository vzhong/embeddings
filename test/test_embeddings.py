from embeddings.embedding import Embedding
import unittest
import os
import shutil


class TestEmbeddings(unittest.TestCase):

    hello_url = 'https://gist.githubusercontent.com/vzhong/c3b439bc165e6c93641bcc569c716c92/raw/87721005f4d9a0e80e9d023aeaf6136c6fee3a72/hello.txt'

    def setUp(self):
        self.root = os.environ['EMBEDDINGS_ROOT'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_root')
        os.makedirs(self.root)
        self.e = Embedding()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_path(self):
        self.assertEqual(os.path.join(self.root, 'foobar'), self.e.path('foobar'))

    def test_download_file(self):
        fname = self.e.download_file(self.hello_url, self.e.path('hello.txt'))
        with open(fname) as f:
            content = f.read()
        self.assertEqual('hello world!', content)

    def test_ensure_file(self):
        self.e.ensure_file('hello_ensure.txt', url=self.hello_url)
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'hello_ensure.txt')))

    def test_initialize_db(self):
        self.e.initialize_db(self.e.path('mydb.db'))
        self.assertTrue(os.path.isfile(os.path.join(self.root, 'mydb.db')))

    def test_insert(self):
        self.e.db = self.e.initialize_db(self.e.path('mydb.db'))
        self.e.insert_batch([
            ('hello', [1, 2, 3]),
            ('world', [2, 3, 4]),
            ('!', [3, 4, 5]),
        ])

        self.assertTrue('world' in self.e)
        self.assertFalse('worlds' in self.e)
        self.assertEqual(3, len(self.e))
        self.assertListEqual([2, 3, 4], self.e.lookup('world'))
        

if __name__ == '__main__':
    unittest.main()
