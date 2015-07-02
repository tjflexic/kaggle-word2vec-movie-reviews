#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pandas as pd

from random import shuffle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""
    
    negators = [line.strip() for line in file('../data/negator.txt','r')]

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Extract special negator like n't
        review_text = re.sub('n\'t', ' not', review_text)
        #
        # 3. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 4. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 5. Optionally remove stop words except for negators (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops or w in KaggleWord2VecUtility.negators]
        #
        # 6. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences

    # Define a function to split a DataFrame for training and test
    @staticmethod
    def split_train_test( df, test_portion=0.3 ):
        # create random list of indices
        N = len(df)
        l = range(N)
        shuffle(l)
 
        # get splitting indicies
        trainLen = int(N*(1-test_portion))
 
        # get training and test sets
        train = df.ix[l[:trainLen]]
        test = df.ix[l[trainLen:]]
 
        return train, test