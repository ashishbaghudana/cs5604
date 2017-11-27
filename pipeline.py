from tokenizers import Tokenizer, WordPunctTokenizer
from mappers import Mapper, LowercaseMapper, PorterStemmer
from filters import Filter, LengthFilter, PunctuationFilter, StopwordFilter

import logging


class Pipeline(object):
    def __init__(self, tokenizer=None, mappers=None, filters=None):
        self.tokenizer = tokenizer
        self.mappers = mappers
        self.filters = filters
        self.logger = logging.getLogger('Pipeline')

        if self.tokenizer is None:
            self.tokenizer = WordPunctTokenizer()

        if self.mappers is None:
            self.mappers = [LowercaseMapper(), PorterStemmer()]

        if self.filters is None:
            self.filters = [StopwordFilter(), PunctuationFilter(), LengthFilter()]

        try:
            assert isinstance(self.tokenizer, Tokenizer)
        except AssertionError:
            raise ValueError('Please provide a valid tokenizer of type infinitygrams.preprocess.tokenizers.Tokenizer')

        try:
            for mapper_obj in self.mappers:
                assert isinstance(mapper_obj, Mapper)
        except AssertionError:
            raise ValueError('Please provide a valid mapper of type infinitygrams.preprocess.tokenizers.Mapper')

        try:
            for filter_obj in self.filters:
                assert isinstance(filter_obj, Filter)
        except AssertionError:
            raise ValueError('Please provide a valid filter of type infinitygrams.preprocess.tokenizers.Filter')

    def preprocess(self, doc):
        tokens = self.tokenizer.tokenize(doc)
        for mapper_obj in self.mappers:
            tokens = map(mapper_obj.map, tokens)
        for filter_obj in self.filters:
            tokens = filter(filter_obj.filter, tokens)
        return tokens


