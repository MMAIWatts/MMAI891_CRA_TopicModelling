# load library
import os
import pandas as pd
import gensim
from gensim import models

from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
import re

from nltk.tokenize import word_tokenize
from language_detector import detect_language

import pkg_resources
from symspellpy import SymSpell, Verbosity

import string
import spacy
from spacy.lang.en import English

from parameters import *


######################################
# -----------Read data----------------#
######################################


def read_data(in_dir, filename, text_col, renamed_text_col, text_language):
    assert len(renamed_text_col) > 0
    df = pd.read_csv(os.path.join(in_dir, filename))
    df.rename(columns={text_col: renamed_text_col}, inplace=True)

    if text_language == 'EN':
        # filter english only and not null rows
        df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'EN'),
                    [renamed_text_col]]
    elif text_language == 'FR':
        df = df.loc[(~df[renamed_text_col].isna()) & (df['Language answered'] == 'FR'),
                    [renamed_text_col]]
    elif text_language == 'ALL':
        df = df.loc[~df[renamed_text_col].isna(), [renamed_text_col]]

    else:
        raise ValueError

    df = df.reset_index()

    return df


######################################
# ---------Text preprocessing---------#
######################################


# Spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


#### Sentence level preprocess ####

# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 4: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)

    # normalization 5: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 6: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are french
    return detect_language(s) in {'English', 'French'}


#### word level preprocess ####
punctuations = string.punctuation

# filtering out punctuations and numbers --maybe it is better to keep years
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    # return [word for word in w_list if word.isalpha()]
    return [word for word in w_list if word not in punctuations]

def f_num(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with number filter out except years
    """
    digits = [num for num in w_list if num.isdigit() and num not in ['2014','2015', '2016', '2017', '2018', '2019']]
    return [word for word in w_list if word not in digits]

# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3,
                                       ignore_token=r"\d{2}\w*\b|\d+\W+\d+\b|\d\w*\b|[!@£#$%^&*();,.?:{}/|<>]")
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# NLTK
# lemmer = WordNetLemmatizer()

def f_lemma(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    # return [lemmer.lemmatize(word) for word in w_list]
    return [word.lemma_.strip() if word.lemma_ != "-PRON-" else word.lower_ for word in w_list]


# filtering out stop words
# create English stop words list
stop_words_to_keep = ["no", "n't", "not"]
stop_words = [stop for stop in spacy.lang.en.stop_words.STOP_WORDS if stop not in stop_words_to_keep]

# stop_words = (list(
#     set(get_stop_words('en'))
#     # | set(get_stop_words('fr'))
# ))
stop_words = spacy.lang.en.stop_words.STOP_WORDS

stop_words = [stop for stop in stop_words if stop not in stop_words_to_keep]
extend_stop_words = EXTRA_STOP_WORDS
stop_words.extend(extend_stop_words)


def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]


# TODO change this function to be more general
def f_replacew(w_list):
    w_list = ['not' if word in ['no', "n't"] else word for word in w_list]
    w_list = ['information' if word == 'info' else word for word in w_list]
    return w_list


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


parser = English()
def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = parser(s)
    w_list = f_lemma(w_list)
    w_list = f_punct(w_list)
    w_list = f_num(w_list)
    w_list = f_typo(w_list)
    w_list = f_stopw(w_list)
    w_list = f_replacew(w_list)

    return w_list


def sent_to_words(sentences):
    for sent in sentences:
        sent = preprocess_sent(sent)
        sent = preprocess_word(sent)

        yield (sent)

    # !python3 -m spacy download en  # run in terminal once


def process_text_col(df, text_col, not_allowed_postags=[]): #allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    # Convert to list
    data = df[text_col].values.tolist()
    texts = list(sent_to_words(data))

    texts = ['vague' if text is None or len(text) == 0 else text for text in texts]

    ##- Build the bigram and trigram models -##
    bigram = gensim.models.Phrases(texts, min_count=2, threshold=10)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_  not in not_allowed_postags])
    return texts_out


if __name__ == '__main__':
    pass
