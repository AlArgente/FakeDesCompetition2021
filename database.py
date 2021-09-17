"""
Class for loading data from multiple file-extensions.
"""
import numpy as np
import pandas as pd

class Database:
    def __init__(self, path):
        self.__path = path # str
        self.__data = None # pd.DataFrame

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    @property
    def path(self) -> str:
        """Path getter
        """
        return self.__path

    def set_path(self, path: str):
        """Path setter
        """
        self.__path = path


    def load_data(self):
        """Function to load the data based on the extension.
        :param train_path: (str) path to train corpus
        :param test_path: (str) path to test_corpus
        """
        functions = {'csv': self.__load_data_csv,
                    'tsv': self.__load_data_csv,
                    'xlsx': self.__load_data_xlsx}

        delimeters = {'csv':',',
            'xlsx': None,
            'tsv': '\t'}

        extension = self.__check_extension()
        func = functions[extension]
        sep = delimeters[extension]
        self.__data = func(sep=sep)

    def __load_data_csv(self, sep=',') -> pd.DataFrame:
        """Function to load data from csv.
        :param train_path: (str) path to train corpus
        :param test_path: (str) path to test corpus
        :param sep: (str) delimiter, if \t, it can read tsv files.
        :return: Pandas.DataFrame, Pandas.DataFrame
        """
        return pd.read_excel(self.__path)

    def __load_data_xlsx(self, sep=None) -> pd.DataFrame:
        """Function to load data from excell.
        :param train_path: (str) path to train corpus
        :param test_path: (str) path to test corpus
        :return: Pandas.DataFrame, Pandas.DataFrame
        """
        return pd.read_excel(self.__path)

    def __check_extension(self) -> str:
        """Function that check the extension of a file. 
        Available extensions: csv, xlsx.
        :return: the extension of the file.
        """
        filename = self.__path
        if filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith('.xlsx'):
            return 'xlsx'
        elif filename.endswith('.tsv'):
            return 'tsv'
        else:
            print('Extension not available right now.')
            return None