from embeddings import GloveEmbeddings, FTEmbeddings, W2VEmbedding

class FactoryEmbeddings:
    """Factory class
    This class will be used to initialize the different embeddings with just one parameter.
    """

    def __init__(self):
        """Sole constructor for the class
        """
        self.__type = None
        self.__embeddings = None

    def embeddings():
        doc = "The embeddings property"
        def fget(self):
            return self.__embeddings
        def fset(self, value):
            self.__embeddings = value
        def fdel(self):
            del self.__embeddings
        return locals()
    embeddings = property(**embeddings())

    def load_embeddings(self, type):
        """Initilize the embeddings from the factory
        Arguments:
            - type (str): Name of the embeddings to use (glove or fasttext).
        """
        self.__type = type.lower()
        if self.__type == 'glove':
            self.__embeddings = GloveEmbeddings()
        elif self.__type == 'fasttext':
            self.__embeddings = FTEmbeddings()
        elif self.__type == 'word2vec':
            self.__embeddings = W2VEmbedding()
        else:
            print('No other embeddings implemented yet.')

