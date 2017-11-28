from abc import ABCMeta, abstractmethod
from nltk.corpus import stopwords

import string


class Filter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def filter(self, token):
        pass


class StopwordFilter(Filter):
    def __init__(self, stop_words=None):
        self.stop_words = stop_words
        if self.stop_words is None:
            self.stop_words = stopwords.words('english')

    def filter(self, token):
        return token not in self.stop_words


class LengthFilter(Filter):
    def __init__(self, length=3):
        self.length = length

    def filter(self, token):
        return len(token) > self.length


class PunctuationFilter(Filter):
    def __init__(self, punctuations=None):
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = set(string.punctuation)

    def filter(self, token):
        return token not in self.punctuations


class WebpageTokensFilter(Filter):
    def __init__(self):
        self.words = {'username', 'password', 'archive', 'live', 'pictures', 'site', 'advertisement', 'skip', 'share',
                      'reuters', 'news', 'cbs', 'npr', 'csbn', 'contact', 'media'}

    def filter(self, token):
        return token not in self.words


def get_filters(filter_list):
    filters = []
    for filter in filter_list:
        filters.append(get_filter(filter))
    return filters


def get_filter(filter):
    if filter == 'stopwordfilter':
        return StopwordFilter()
    elif filter == 'punctuationfilter':
        return PunctuationFilter()
    elif filter == 'webpagetokensfilter':
        return WebpageTokensFilter()
