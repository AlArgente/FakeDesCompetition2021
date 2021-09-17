import io
import time
import numpy as np
from abc import ABC, abstractmethod
from nltk import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors

class Embeddings(ABC):
    """Abstract class to load different embeddings
    """
    def __init__(self):
        self.d = 300
        self.__vocabulary = {}
        self.__embeddings_matrix = []
        self.embeddings = {}

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def embeddings_matrix(self):
        return self.__embeddings_matrix

    @property
    def embeddings_full(self):
        return self.embeddings

    @abstractmethod
    def load_vectors(self, fname):
        pass

    @abstractmethod
    def load_embeddings(self, fname):
        pass

    def calc_embeddings(self, text, max_sequence_len, max_sentences_in_sequence):
        """Function that apply the vector to get the embeddings from the text.
        Arguments:
            - text: list of lists with the text at least tokenized.
        Returns:
            - embeddings: list of embeddings.
        """
        # This function will be deleted in the future
        embeddings = []
        null_embeddings = np.zeros(self.d)
        for sequence in text:
            max_words_allowed = 0
            sentences = sent_tokenize(sequence)
            embedding_sequence = null_embeddings
            count = 0 # Count all the lines
            for sentence in sentences:
                if count >= max_sentences_in_sequence: # If limit of sentences reached
                    continue
                embedding_sentence = null_embeddings
                words = word_tokenize(sentence, preserve_line=True)
                count_word = 0
                count += 1
                for word in words: # Every word in the line
                    count_word += 1
                    if max_words_allowed >= max_sequence_len:
                        continue
                    if word in self.embeddings.keys():
                        embedding_word = self.embeddings[word]
                    else:
                        # embedding_word = np.random.normal(0, 1, self.d)
                        embedding_word = null_embeddings
                    embedding_sentence = np.sum([embedding_sentence, embedding_word], axis=0)
                    max_words_allowed+=1
                # embedding_sentence = embedding_sentence / count_word
                embedding_sequence = np.sum([embedding_sequence, embedding_sentence], axis=0)
            embedding_sequence = embedding_sequence / count # Divide by all the lines
            embeddings.append(embedding_sequence)

        """
        # Alternative:
        for i in range(len(text)):
            embedding_sentence = []
            for j in range(len(text[i])):
                if text[i][j] in self.data:
                    embedding_word = self.data[text[i][j]]
                else:
                    embedding_word = np.random.normal(0, 1, self.d)
                embedding_sentence.append(embedding_word)
            embeddings.append(embedding_sentence)
        """
        if len(text) != len(embeddings):
            raise ValueError(
                "The len of the text and the embeddings corresponding to it, arent the same."
            )
        return np.array(embeddings)



class GloveEmbeddings(Embeddings):
    """Class to load the Glove embeddings
    """
    def __init__(self):
        super(GloveEmbeddings, self).__init__()
        # self.load_embeddings()
        self.load_vectors()
        print('Embeddings cargados')

    def load_vectors(self, fname='../glove.6B.300d.txt'):
        """Function to load the Glove embeddings instead of random initialize them
        """
        print('Loading Glove')
        # Aux word for possibles new words out of our vocabulary and for padding items
        # self.vocabulary['PADDING'] = len(self.vocabulary)
        self.vocabulary['UNKNOWN'] = len(self.vocabulary)
        # Generate a vector of 0 for padding items
        # self.embeddings_matrix.append(np.zeros(300))
        # Generate a random embedding for this new words.
        self.embeddings_matrix.append(np.random.normal(0, 1, 300))
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                token = line.split(' ')
                # Read the embedding as a np.array
                vec = np.array(token[1:], dtype=np.float32)
                # Create the vocabulary with Glove embeddings
                # Adding to each word the corresponding index
                self.vocabulary[token[0]] = len(self.vocabulary)
                # Add the embedding to the matrix of embeddings
                self.embeddings_matrix.append(vec)

    def load_embeddings(self, fname='../glove.6B.300d.txt'):
        print('Loading Glove Embeddings')
        # Open the Glove file
        with open(fname, 'r') as f:
            # Read all the lines
            lines = f.readlines()
            # Create the embeddings
            for line in lines:
                token = line.split(' ')
                self.embeddings[token[0]] = np.array(token[1:], dtype=np.float32)



class FTEmbeddings(Embeddings):
    """Class to load the FastText embeddings
    """
    def __init__(self):
        super(FTEmbeddings, self).__init__()
        self.load_embeddings()
        print('Embeddings cargados')

    def load_vectors(self, fname='../wiki-news-300d-1M.vec'):
        """Function to load the Fasttext embeddings instead of random initialize them
        """
        # To probe with 2M words instead of 1M
        # fname = '../crawl-300d-2M.vec'
        fname = './embeddings-l-model.vec'
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        print('Loading FastText')
        # Aux word for possibles new words out of our vocabulary
        self.vocabulary['NEWWORD'] = len(self.vocabulary)
        # Generate a random embedding for this new words.
        self.embeddings_matrix.append(np.random.normal(0, 1, 300))
        for line in fin:
            tokens = line.rstrip().split(' ')
            if (len(tokens[1:]) == 300):
                # data[tokens[0]] = map(float, tokens[1:])
                # Create the vocabulary with FastText embeddings
                # Adding to each word the corresponding index
                self.vocabulary[tokens[0]] = len(self.vocabulary)
                # Add the embedding to the matrix of embeddings as a np.array
                self.embeddings_matrix.append(np.array(tokens[1:]))

    def load_embeddings(self, fname='../wiki-news-300d-1M.vec'):
        # fname = '/home/alberto/Escritorio/TFM/wiki-news-300d-1M.vec'
        fname = './embeddings-l-model.vec'
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        print('Loading FastText Embeddings.')
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.embeddings[tokens[0]] = np.array(tokens[1:], dtype=np.float32)



class W2VEmbedding(Embeddings):
    """Class to load the Word2Vec embeddings
    """
    def __init__(self):
        super(W2VEmbedding, self).__init__()
        self.load_embeddings()
        print('Embeddings cargados')

    def load_vectors(self, fname='../GoogleNews-vectors-negative300.bin'):
        print('Loading Word2Vec Embeddings.')
        w2v_model = KeyedVectors.load_word2vec_format(fname, binary=True)
        # Aux word for possibles new words out of our vocabulary
        self.vocabulary['NEWWORD'] = len(self.vocabulary)
        # Generate a random embedding for this new words.
        self.embeddings_matrix.append(np.random.normal(0, 1, 300))
        for token in w2v_model.vocab:
            # Create the vocabulary with FastText embeddings
            # Adding to each word the corresponding index
            self.vocabulary[token] = len(self.vocabulary)
            # Add the embedding to the matrix of embeddings
            self.embeddings_matrix.append(w2v_model[token])

    def load_embeddings(self, fname='../GoogleNews-vectors-negative300.bin'):
        print('Loading Word2Vec Embeddings.')
        w2v_model = KeyedVectors.load_word2vec_format(fname, binary=True)
        for token in w2v_model.vocab:
            self.embeddings[token] = w2v_model[token]