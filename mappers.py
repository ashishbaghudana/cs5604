from abc import ABCMeta, abstractmethod
from nltk import stem


class Mapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, token):
        pass


class Stemmer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def stem(self, token):
        pass


class Lemmatizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def lemmatize(self, token):
        pass


class LowercaseMapper(Mapper):
    def map(self, token):
        return token.lower()


class WordNetLemmatizer(Lemmatizer, Mapper):
    def __init__(self):
        self.lemmatizer = stem.WordNetLemmatizer()

    def lemmatize(self, token):
        return self.lemmatizer.lemmatize(token)

    def map(self, token):
        return self.lemmatize(token)


class PorterStemmer(Stemmer, Mapper):
    def __init__(self):
        self.stemmer = stem.PorterStemmer()

    def stem(self, token):
        return self.stemmer.stem(token)

    def map(self, token):
        return self.stem(token)


def get_mappers(mapper_list):
    mappers = []
    for mapper in mapper_list:
        mappers.append(get_mapper(mapper))
    return mappers


def get_mapper(mapper):
    if mapper == 'lowercasemapper':
        return LowercaseMapper()
    elif mapper == 'wordnetlemmatizer':
        return WordNetLemmatizer()
    elif mapper == 'porterstemmer':
        return PorterStemmer()
    else:
        return None
