import logging
import progressbar
import mmap

from hbase import Database


class WebpageTokensReader(object):
    def __init__(self, filename, pipeline=None):
        self.filename = filename
        self.pipeline = pipeline
        self.num_docs = _raw_in_count(self.filename)
        self.documents = {}
        self.preprocess()

    def preprocess(self):
        logging.info('Start preprocessing all documents')
        bar = progressbar.ProgressBar(maxval=self.num_docs)
        with open(self.filename, 'r') as freader:
            if self.pipeline is None:
                for line in freader:
                    self.documents[line.split('\t')[0].strip()] = line.split('\t')[1].strip().split(',')
                    bar.update(bar.currval + 1)
            else:
                logging.info('Using the preprocessing pipeline to process each document')
                for line in freader:
                    self.documents[line.split('\t')[0].strip()] = self.pipeline.preprocess(line.split('\t')[1].strip())
                    bar.update(bar.currval + 1)

    def __iter__(self):
        for document in self.documents.values():
            yield document

    def __len__(self):
        return len(self.documents)

    def items(self):
        for key, value in self.documents.items():
            yield key, value


class HBaseReader(object):
    def __init__(self, table_name, collection_name, pipeline=None):
        self.database = Database()
        self.documents = {}
        self.table_name = table_name
        self.collection_name = collection_name
        self.pipeline = pipeline
        self.preprocess()

    def preprocess(self):
        documents = self.database.get_collection_webpages(self.table_name, self.collection_name)
        if self.pipeline is not None:
            for _id, doc in documents.items():
                self.documents[_id] = self.pipeline.preprocess(doc)
        else:
            self.documents = documents

    def __iter__(self):
        for document in self.documents.values():
            yield document

    def items(self):
        for key, value in self.documents.items():
            yield key, value

    def __len__(self):
        return len(self.documents)


def _raw_in_count(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines
