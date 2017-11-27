from hbase import Database


class WebpageTokensReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.documents = []
        self.preprocess()

    def preprocess(self):
        with open(self.filename, 'r') as freader:
            for line in freader:
                self.documents.append(line.strip().split(','))

    def __iter__(self):
        for document in self.documents:
            yield document

    def __len__(self):
        return len(self.documents)


class HBaseReader(object):
    def __init__(self, table_name, collection_name, pipeline=None):
        self.database = Database()
        self.documents = []
        self.table_name = table_name
        self.collection_name = collection_name
        self.pipeline = pipeline

    def preprocess(self):
        documents = self.database.get_collection_tweets(self.table_name, self.collection_name)
        if self.pipeline is not None:
            for doc in documents:
                self.documents.append(self.pipeline.preprocess(doc))
        else:
            self.documents = documents

    def __iter__(self):
        for document in self.documents:
            yield document

    def __len__(self):
        return len(self.documents)

