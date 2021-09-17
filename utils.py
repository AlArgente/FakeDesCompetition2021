"""
File of utils functions.
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import ngrams
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import spacy
from sklearn import metrics
from nela_features.nela_features import NELAFeatureExtractor


def load_data_by_extension(train_path=None, test_path=None):
    """Function to load the data based on the extension.
    :param train_path: (str) path to train corpus
    :param test_path: (str) path to test_corpus
    :return: Pandas.DataFrame, Pandas.DataFrame
    """
    functions = {'csv': load_data_csv,
                 'tsv': load_data_csv,
                 'xlsx': load_data_xlsx}

    delimeters = {'csv':',',
           'xlsx': None,
           'tsv': '\t'}

    extension = check_extension(train_path)
    func = functions[extension]
    sep = delimeters[extension]
    return func(train_path=train_path, test_path=test_path, sep=sep)

def load_data_csv(train_path=None, test_path=None, sep=','):
    """Function to load data from csv.
    :param train_path: (str) path to train corpus
    :param test_path: (str) path to test corpus
    :param sep: (str) delimiter, if \t, it can read tsv files.
    :return: Pandas.DataFrame, Pandas.DataFrame
    """
    assert train_path is not None

    test = None
    train = pd.read_csv(train_path, sep=sep)

    if test_path is not None:
        test = pd.read_csv(test_path, sep=sep)

    return train, test

def load_data_xlsx(train_path=None, test_path=None, sep=None):
    """Function to load data from excell.
    :param train_path: (str) path to train corpus
    :param test_path: (str) path to test corpus
    :return: Pandas.DataFrame, Pandas.DataFrame
    """
    assert train_path is not None

    test = None
    train = pd.read_excel(train_path, engine='openpyxl')

    if test_path is not None:
        test = pd.read_excel(test_path, engine='openpyxl')

    return train, test


def tokenize(text: list, language='es') -> list:
    """Function that tokenize the text and remove stopwords.
    Here I use spacy tokenizer so I can remove stopwords with the tokenizer.
    Arguments:
        - text: text to be tokenize
    Returns:
        - A list with all the text tokenized
    """
    spacy_tokenizer = spacy.load(language)
    return [[word.lower_ for word in spacy_tokenizer(comment) if word.is_stop is False] for comment in text]


def remove_stopwords(text: list, language='es') -> list:
    """Function to delete the stopwords in english
    Arguments:
        - text: text to delete the stopwords. The sentences must be tokenized first.
        - langueage: langueage to remove stopwords. Languages avaiable: English and Spanish.
    Returns:
        - text without stopwords
    """
    languages = {'en': 'english', 'es': 'spanish'}
    stop_words_l = get_stop_words(language=language)
    nltk_stopwords = stopwords.words(languages[language])
    all_stopwords = stop_words_l + nltk_stopwords

    return [[w for w in word if w not in all_stopwords] for word in text]


def generate_ngrams(text: list, n=2) -> list:
    """Function that generate ngrams based on param
    :param text: list of strings; text to generate ngrams
    :param n: number for n-grams
    :return: All ngrams
    """
    return [list(ngrams(comment, n)) for comment in text]


def split_train_test(data, test_size=0.2):
    """Function that split train_test data based on param
    :param data: data to split
    :param test_size: (float) length for test data
    :return: train data, test data
    """
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test


def stemming(text):
    """Function to get the stem for every word
    Arguments:
        - text: list of lists with the text tokenized.
    Returns:
        - list of lists with the stem applied.
    """
    stemmer = PorterStemmer()
    # Stem all the data
    stem_list = [[stemmer.stem(token) for token in comment] for comment in text]
    # Generate the instances with join
    stem_join = [' '.join(comment) for comment in stem_list]
    return stem_list, stem_join


def tfidf(train_text, dev_text, use_ngrams, n_grams, type_n_grams):
    """Function that generate Tf-Idf
    :param train_text: (list) train data, must be at lesat tokenized
    :param dev_text: (list) test data, must be at least tokenized
    :param use_ngrams: (bool) True/False to use or not Ngrams
    :param n_grams: (int) number for n-grams
    :return: tfidf feature for train and test.
    """
    assert isinstance(train_text, list)
    assert len(train_text) > 0
    assert len(dev_text) > 0
    assert isinstance(dev_text, list)
    assert isinstance(use_ngrams, bool)
    assert isinstance(use_ngrams, bool) and n_grams > 1
    assert isinstance(n_grams, int)

    if not use_ngrams:
        vectorizer = TfidfVectorizer(analyzer=type_n_grams, tokenizer=lambda x: x, preprocessor=lambda x: x,
                                     lowercase=True)
    else:
        vectorizer = TfidfVectorizer(analyzer=type_n_grams, tokenizer=lambda x: x, preprocessor=lambda x: x,
                                     lowercase=True, ngram_range=(1, n_grams))
    train_text = np.array(train_text)
    dev_text = np.array(dev_text)

    train_tfidf = vectorizer.fit_transform(train_text)
    dev_tfidf = vectorizer.transform(dev_text)

    return train_tfidf.toarray(), dev_tfidf.toarray(), vectorizer

def extract_nela(text: list) -> list:
    """Function that extrall all nela features
    :param text: (list) text to extrall all nela features
    :return: nela features for every input comment
    """
    nela = NELAFeatureExtractor()
    return [nela.extract_all(comment)[0] for comment in text]

# METRICS
def acc_f1macro_creport(y_true: list, y_pred: list):
    """Function that calculate accuracy, f1_macro and classification report
    :param y_true: real annotation
    :param y_pred: predicted annotation
    :return: accuracy, f1_macro, classification_report
    """
    assert len(y_true) > 0
    assert len(y_pred) > 0
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred), metrics.f1_score(y_true=y_true, y_pred=y_pred,
                                                                                  average='macro'), \
           metrics.classification_report(y_true=y_true, y_pred=y_pred)


def accurcy_score(y_true: list, y_pred: list) -> float:
    """Function that calculate accuracy
    :param y_true: real annotation
    :param y_pred: predicted annotation
    :return: accuracy
    """
    assert len(y_true) > 0
    assert len(y_pred) > 0
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def f1_score(y_true: list, y_pred: list, average='macro', pos_label='Fake') -> float:
    """Function that calculate f1
    :param y_true: real annotation
    :param y_pred: predicted annotation
    :param average: 'micro', 'macro'
    :return: f1
    """
    assert len(y_true) > 0
    assert len(y_pred) > 0
    if average=='macro':
        return metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)
    elif average=='binary':
        return metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average, pos_label=pos_label)


def classification_report(y_true: list, y_pred: list) -> str:
    """Function that calculate classification report
    :param y_true: real annotation
    :param y_pred: predicted annotation
    :return str: classification report from scikit-learn. 
    """
    assert len(y_true) > 0
    assert len(y_pred) > 0
    return metrics.classification_report(y_true=y_true, y_pred=y_pred)


def check_extension(filename: str) -> str:
    """Function that check the extension of a file. 
    Available extensions: csv, xlsx.
    :param filename: filename to check
    :return: the extension of the file.
    """
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.tsv'):
        return 'tsv'
    else:
        print('Extension not available right now.')
        return None