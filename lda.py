from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from readers import WebpageTokensReader, HBaseReader
from mappers import get_mappers
from filters import get_filters
from tokenizers import get_tokenizer
from pipeline import Pipeline
from tabulate import tabulate
from pyLDAvis.gensim import _extract_data, prepare
from pyLDAvis import save_html
from timeit import default_timer as timer
from datetime import timedelta

import os
import sys
import logging
import argparse
import numpy as np
import progressbar


class Constants(object):
    SAVE_DIR = '/home/cs5604f17_cta/models/{}'
    SAVE_FILE_FORMAT = '{}_topics_{}_alpha_{}_beta_{}_iterations_{}.model'
    SAVE_HTML_FORMAT = '{}_topics_{}_alpha_{}_beta_{}_iterations_{}.html'
    SAVE_TOPIC_KEYWORDS = '{}_topics_keywords_{}_alpha_{}_beta_{}_iterations_{}.txt'
    SAVE_DOCUMENT_TOPICS = '{}_document_topics_{}_alpha_{}_beta_{}_iterations_{}.txt'
    SAVE_DOCUMENT_KEYWORDS = '{}_document_keywords_{}_alpha_{}_beta_{}_iterations_{}.txt'


class Corpus(object):
    def __init__(self, documents, dictionary):
        self.documents = documents
        self.dictionary = dictionary

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        for document in self.documents:
            yield self.dictionary.doc2bow(document)


class LDA(object):
    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary

    def run_model(self, collection_name, num_topics, save_dir=None, save_file=None, alpha=0.1, beta=0.01,
                  iterations=800, passes=1):
        model = LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, alpha=alpha, eta=beta,
                             iterations=iterations, passes=passes)
        if save_dir is None:
            save_dir = Constants.SAVE_DIR.format(collection_name.lower().replace(' ', '_'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if save_file is None:
            save_file = Constants.SAVE_FILE_FORMAT.format(collection_name.lower().replace(' ', '_'), num_topics, alpha,
                                                          beta, iterations)
        logging.info(save_dir)
        model.save(os.path.join(save_dir, save_file))
        return model

    def run_models(self, collection_name, topics_range, save_dir=None, alpha=0.1, beta=0.01, iterations=800):
        logging.info('Starting runs of topic modelling')
        perplexity = []
        coherence = []
        for num_topics in topics_range:
            logging.info('Running topic model on num_topics = %s' % num_topics)
            start = timer()
            model = self.run_model(collection_name, num_topics, save_dir=save_dir, save_file=None, alpha=alpha,
                                   beta=beta, iterations=iterations)
            end = timer()
            logging.info('Time taken to create model = %s min' % str(timedelta(seconds=end-start)))

            start = timer()
            perp = self.check_perplexity(model)
            end = timer()
            logging.info('Perplexity for the model = %s' % perp)
            logging.info('Time taken to calculate perplexity = %s' % str(timedelta(seconds=end-start)))
            perplexity.append(perp)

            start = timer()
            coher = self.check_topic_coherence(model)
            end = timer()
            logging.info('Coherence for the model = %s' % coher)
            logging.info('Time taken to calculate topic coherence = %s' % str(timedelta(seconds=end-start)))
            coherence.append(coher)

            save_file_document_topics = os.path.join(save_dir, Constants.SAVE_DOCUMENT_TOPICS.format(
                collection_name.lower().replace(' ', '_'), num_topics, alpha, beta, iterations))
            save_file_document_keywords = os.path.join(save_dir, Constants.SAVE_DOCUMENT_KEYWORDS.format(
                collection_name.lower().replace(' ', '_'), num_topics, alpha, beta, iterations))
            save_file_topic_keywords = os.path.join(save_dir, Constants.SAVE_TOPIC_KEYWORDS.format(
                collection_name.lower().replace(' ', '_'), num_topics, alpha, beta, iterations))

            data = _extract_data(model, self.corpus, self.dictionary)
            prepared_data = prepare(topic_model=model, corpus=self.corpus, dictionary=self.dictionary)
            save_file = Constants.SAVE_HTML_FORMAT.format(collection_name.lower().replace(' ', '_'), num_topics, alpha,
                                                          beta, iterations)
            with open(os.path.join(save_dir, save_file), 'w') as fwriter:
                save_html(prepared_data, fwriter)

            topic_labels, topic_tokens = self.get_topic_keywords(model, save_file_topic_keywords)
            self.get_document_keywords(model, save_file_document_topics, save_file_document_keywords, data,
                                       topic_labels, topic_tokens)
        return perplexity, coherence

    def get_document_keywords(self, model, save_file_document_topics, save_file_document_keywords, data,
                              topic_labels, topic_tokens, top_topics=3, top_n=40):
        # document_keywords = {}
        document_topics = {}
        document_labels = {}
        document_topics_probabilities = {}

        topic_probabilities = data['doc_topic_dists']
        logging.info('Getting words for documents')
        bar = progressbar.ProgressBar(max_value=self.corpus.documents.num_docs)
        idx = 0
        for _id, doc in bar(enumerate(self.corpus.documents.items())):
            topic_idx = np.argpartition(data['doc_topic_dists'][idx], -top_topics)[-top_topics:]
            topic_prob = [topic_probabilities[_id][topic_id] for topic_id in topic_idx]
            top_words = []
            top_labels = []
            for topic_id in topic_idx:
                top_words.append(topic_tokens[topic_id])
                top_labels.append(topic_labels[topic_id])
            # document_keywords[_id] = ','.join(top_words)
            document_topics[_id] = ','.join([str(i) for i in topic_idx])
            document_topics_probabilities[_id] = ','.join([str(i) for i in topic_prob])
            document_labels[_id] = ','.join([str(i) for i in top_labels])

            # logging.info('Processed docs: %d' % idx)
            # idx += 1

        with open(save_file_document_topics, 'w') as fwriter:
            idx = 0
            for _id, document in self.corpus.documents.items():
                fwriter.write(_id + '\t' + document_labels[idx] + '\t' + document_topics[idx] + '\t'
                              + document_topics_probabilities[idx] + '\n')
                idx += 1
        # with open(save_file_document_keywords, 'w') as fwriter:
        #     for _id, document in document_keywords.items():
        #         fwriter.write(str(_id) + '\t' + document + '\n')

    def get_topic_keywords(self, model, save_file_topic_keywords, topn=40):
        topic_labels = []
        topic_labels_set = set()
        topic_tokens = []

        top_terms = []
        for topic in range(model.num_topics):
            topic_top_terms = np.array(model.show_topic(topicid=topic, topn=topn))[:, 0]
            top_terms.append(topic_top_terms)
            top_tokens = ','.join(topic_top_terms)
            topic_tokens.append(top_tokens)

            idx = 0
            while topic_top_terms[idx] in topic_labels_set:
                idx += 1
            topic_labels.append(topic_top_terms[idx])
            topic_labels_set.add(topic_top_terms[idx])

        with open(save_file_topic_keywords, 'w') as fwriter:
            idx = 0
            for label, topic in zip(topic_labels, topic_tokens):
                fwriter.write(str(idx) + '\t' + label + '\t' + topic + '\n')
                idx += 1

        return topic_labels, topic_tokens

    def check_perplexity(self, model):
        return model.log_perplexity(self.corpus)

    def check_topic_coherence(self, model, coherence='u_mass'):
        coherence_model = CoherenceModel(model, dictionary=self.dictionary, corpus=self.corpus, coherence=coherence)
        return coherence_model.get_coherence()


def main():
    parser = argparse.ArgumentParser(description='Run LDA on a given collection')

    # Required arguments
    required_args_parser = parser.add_argument_group('required arguments')
    required_args_parser.add_argument('-c', '--collection_name', help='Collection name', required=True)
    required_args_parser.add_argument('-t', '--topics', nargs='+', help='Number of topics to run the model on',
                                      required=True, type=int)
    required_args_parser.add_argument('-p', '--preprocess', help='Preprocess data', action='store_true', default=False,
                                      required=False)
    required_args_parser.add_argument('--table_name', help='Table name for HBase', required=False)
    sourcefile = required_args_parser.add_mutually_exclusive_group(required=True)
    sourcefile.add_argument('-hb', '--hbase', help='Get collection from HBase', action='store_true')
    sourcefile.add_argument('-f', '--file', help='File name for tokens')

    # Optional arguments with defaults
    parser.add_argument('-l', '--logs', help='Log directory', default='/tmp/logs')
    parser.add_argument('-a', '--alpha', help='Alpha hyperparameter', default=0.1, type=float)
    parser.add_argument('-b', '--beta', help='Beta hyperparameter', default=0.01, type=float)
    parser.add_argument('-i', '--iter', help='Number of iterations', default=800, type=int)
    parser.add_argument('--save_dir', help='Save directory for topic models', default=None)

    # Preprocessing arguments with defaults
    preprocess_parser = parser.add_argument_group('preprocessing arguments to be added when using -p flag')
    preprocess_parser.add_argument('--tokenizer', help='Tokenizer to use', default='wordtokenizer')
    preprocess_parser.add_argument('--mappers', help='Mappers to use', nargs='+', default=['lowercasemapper'])
    preprocess_parser.add_argument('--filters', help='Filters to use', nargs='+', default=['stopwordfilter',
                                                                                           'punctuationfilter'])
    preprocess_parser.add_argument('--filter_words', help='Filename with words to filter out', default=None)

    args = parser.parse_args()

    # Arguments for preprocessing pipeline

    logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger('LDA')
    logger.info('Application is starting!')

    if args.preprocess:
        if args.tokenizer is not None and args.mappers is not None and args.filters is not None:
            pipeline = Pipeline(tokenizer=get_tokenizer(args.tokenizer), mappers=get_mappers(args.mappers),
                                filters=get_filters(args.filters, args.filter_words))
        else:
            logger.critical('Cannot preprocess data if the type of tokenizer, mappers and filters is not given')
            sys.exit(2)
    else:
        pipeline = None

    start = timer()
    if args.hbase:
        documents = HBaseReader(args.table_name, args.collection_name, pipeline=pipeline)
    else:
        documents = WebpageTokensReader(args.file, pipeline=pipeline)
    end = timer()
    logging.info('Preprocessing time = %s' % str(timedelta(seconds=end-start)))

    start = timer()
    dictionary = Dictionary(documents)
    corpus = Corpus(documents, dictionary)
    end = timer()
    logging.info('Time taken to build vocabulary = %s' % str(timedelta(seconds=end-start)))

    lda = LDA(corpus, dictionary)
    perplexity, coherence = lda.run_models(args.collection_name, args.topics, args.save_dir, alpha=args.alpha,
                                           beta=args.beta, iterations=args.iter)
    topics = ['Num Topics'] + args.topics
    perplexities = ['Log Perplexity'] + perplexity
    coherences = ['Coherence (UMass)'] + coherence
    table = [topics, perplexities, coherences]
    print (tabulate(table))
    with open('results_{}.txt'.format(args.collection_name.lower().replace(' ', '_')), 'w') as fwriter:
        fwriter.write(tabulate(table))


if __name__ == '__main__':
    main()
