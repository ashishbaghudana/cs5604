from abc import ABCMeta, abstractmethod
from nltk import word_tokenize, wordpunct_tokenize

from nltk.tokenize import TweetTokenizer as TT


class Tokenizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def tokenize(self, doc):
        pass


class SpaceTokenizer(Tokenizer):
    def tokenize(self, doc):
        return doc.split()


class WordTokenizer(Tokenizer):
    def tokenize(self, doc):
        return word_tokenize(doc)


class WordPunctTokenizer(Tokenizer):
    def tokenize(self, doc):
        return wordpunct_tokenize(doc)


class TweetTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = TT()

    def tokenize(self, doc):
        return self.tokenizer.tokenize(doc)


class CommaTokenizer(Tokenizer):
    def tokenize(self, doc):
        return doc.split(',')


class NoOpTokenizer(Tokenizer):
    def tokenize(self, doc):
        return doc


def get_tokenizer(tokenizer):
    if tokenizer == 'spacetokenizer':
        return SpaceTokenizer()
    elif tokenizer == 'wordtokenizer':
        return WordTokenizer()
    elif tokenizer == 'wordpuncttokenizer':
        return WordPunctTokenizer()
    elif tokenizer == 'tweettokenizer':
        return TweetTokenizer()
    elif tokenizer == 'commatokenizer':
        return CommaTokenizer()
    elif tokenizer == 'nooptokenizer':
        return NoOpTokenizer()
    else:
        return WordTokenizer()
