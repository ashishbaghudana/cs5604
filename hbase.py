import happybase
import progressbar
from utils import raw_in_count
import numpy as np


class Database(object):
    def __init__(self, host='localhost', port=9090):
        self.host = host
        self.port = port

    def get_collection_webpages(self, table_name, collection_name, output_file=None):
        connection = happybase.Connection(self.host, self.port)
        table = connection.table(table_name)
        rows = {}
        documents = []
        for row in table.scan():
            if len(row) >= 2 and 'metadata:collection-name' in row[1] and 'metadata:doc-type' in row[1] and \
                            row[1]['metadata:doc-type'] == 'webpage' and \
                            row[1]['metadata:collection-name'] == collection_name:
                rows[row[0]] = (row[1]['clean-webpage:tokens'].split(','))
                documents.append(row[0] + '\t' + row[1]['clean-webpage:tokens'])
        if output_file is not None:
            with open(output_file, 'w') as fwriter:
                for document in documents:
                    fwriter.write(document + '\n')
        return rows

    def get_collection_tweets(self, table_name, collection_name, output_file):
        connection = happybase.Connection(self.host, self.port)
        table = connection.table(table_name)
        rows = []
        documents = []
        for row in table.scan():
            if len(row) >= 2 and 'metadata:collection-name' in row[1] and 'metadata:doc-type' in row[1] and \
                            row[1]['metadata:doc-type'] == 'tweet' and \
                            row[1]['metadata:collection-name'] == collection_name:
                rows.append(row[1]['clean-tweet:clean_tokens'].split(';'))
                documents.append(row[0] + '\t' + row[1]['clean-tweet:clean_tokens'])
        if output_file is not None:
            with open(output_file, 'w') as fwriter:
                for document in documents:
                    fwriter.write(document + '\n')
        return rows

    def batch_upload(self, table_name, document_topics_file, topic_names_file=None, batch_size=20000):
        connection = happybase.Connection(self.host, self.port)
        table = connection.table(table_name)

        topic_names = {}
        if topic_names_file is not None:
            with open(topic_names_file) as freader:
                for line in freader:
                    topic_names[int(line.strip().split('\t')[0])] = line.strip().split('\t')[1]
        bar = progressbar.ProgressBar(max_value=raw_in_count(document_topics_file))
        with table.batch(batch_size=batch_size):
            with open(document_topics_file) as freader:
                for line in bar(freader):
                    _id, topics, topic_ids, topic_prob = line.strip().split('\t')
                    if topic_names_file is None:
                        row = table.row(_id)
                        topic_array = topics.split(',')
                        topic_probability_array = [float(f) for f in topic_prob.split(',')]
                        top_topic = topic_array[np.argmax(topic_probability_array)]
                        row.update({'topic:topic-list': topics, 'topic:probability-list': topic_prob, 'topic:topic-displaynames': top_topic})
                        table.put(_id, row)
                    else:
                        row = table.row(_id)
                        topics = []
                        for topic_id in topic_ids.split(','):
                            topics.append(topic_names[int(topic_id)])
                        topic_probability_array = [float(f) for f in topic_prob.split(',')]
                        top_topic = topics[np.argmax(topic_probability_array)]
                        # print topic_prob, np.argmax(topic_prob), top_topic, topics; break
                        row.update({'topic:topic-list': ','.join(topics), 'topic:probability-list': topic_prob, 'topic:topic-displaynames': top_topic})
                        table.put(_id, row)
