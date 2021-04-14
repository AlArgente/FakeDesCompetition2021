import os
import numpy as np
from enum import Enum
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, KFold
import xgboost as xgb
from utils import *

class ModelSelection(Enum):
    SVM = SVC
    BAYES = MultinomialNB
    RF = RandomForestClassifier
    XGB = xgb.XGBClassifier
    DT = DecisionTreeClassifier

class ModelConfig(Enum):
    SVM = {'C':1, 'kernel': 'rbf','coef0':5, 'degree':6, 'gamma': 'auto', 'max_iter':100,
           'class_weight': 'balanced'}
    BAYES = {'alpha': 0.25, 'fit_prior':True, 'class_prior': None}
    RF = {'n_estimators': 100, 'criterion':'entropy', 'max_depth': 4, 'class_weight': 'balanced'}
    XGB = {'random_state': 42, 'seed':np.random.randint(0, 99999999), 'max_depth':5}
    DT = {'max_depth':50, 'criterion':'entropy', 'class_weight': 'balanced'}

class MlModel:
    def __init__(self, train_path, test_path=None, mlm_name='SVM', language='es', use_ngrams=False, ngrams=1,
                 use_nela_features = False, char_ngram = False):
        self.__train_path = train_path
        self.__test_path = test_path
        self.__language = language
        self.__use_ngrams = use_ngrams
        self.__ngrams = ngrams
        self.__use_nela_features = use_nela_features
        self.__char_ngram = 'char' if char_ngram is True else 'word'
        if mlm_name in ModelSelection.__members__.keys() and mlm_name in ModelConfig.__members__.keys():
            self.__mlm_name = mlm_name
            self.__choose_model(mlm_name)
        else:
            print(ModelSelection.__members__.keys())
            print('The model is not included. Try with: ' + str(
                ', '.join([e.name for e in ModelSelection])
            ))
            print('Please use the set_model(model_name) function to select the model')
            self.__model = None

    # Pipelines

    def pipeline(self):
        self.__print_configuration()
        print('Loading data')
        self.__load_data()
        print('Preparing input data')
        self.__prepare_input_data()
        print('Fitting the model')
        self.__fit_model()
        print('Predicting on test data')
        self.__predict()

    def pipeline2(self):
        self.__print_configuration()
        print('Loading data')
        self.__load_data()
        print('Preparing input data')
        self.__prepare_input_data()
        print('Fitting the model with Cross Validation')
        self.__cv()

    # PRIVATE FUNCTIONS

    def __print_configuration(self):
        print('Configuration:')
        print('Language: {}'.format(self.__language))
        print('Using n_grams: {}'.format(self.__use_ngrams))
        if self.__use_ngrams:
            print('Ngrams to use: {}'.format(self.__ngrams))
        print('Model to use: {}'.format(self.__mlm_name))
        print('Using nela features: {}'.format(self.__use_nela_features))

    def __load_data(self):
        self.__train, self.__test = load_data_xlsx(self.__train_path, self.__test_path)
        if self.__test is None:
            self.__train, self.__test = split_train_test(self.__train)

    def __prepare_input_data(self):
        # Tokenize text
        train_text = tokenize(self.__train['Text'], language=self.__language)
        test_text = tokenize(self.__test['Text'], language=self.__language)

        # Delete stopwords
        train_text = remove_stopwords(train_text, language='es')
        test_text = remove_stopwords(test_text, language='es')

        # Stem text
        train_text, _ = stemming(train_text)
        test_text, _ = stemming(test_text)

        # Tf-Idf
        if self.__char_ngram == 'char':
            train_tfidf, test_tfidf = tfidf(train_text=self.__train['Text'].to_list(), test_text=self.__test['Text'].to_list(),
                                        use_ngrams=self.__use_ngrams, n_grams=self.__ngrams,
                                        type_n_grams=self.__char_ngram)
        else:
            train_tfidf, test_tfidf = tfidf(train_text=train_text, test_text=test_text,
                                        use_ngrams=self.__use_ngrams, n_grams=self.__ngrams,
                                        type_n_grams=self.__char_ngram)

        # Nela Features
        if self.__use_nela_features:
            train_nela = np.array(extract_nela(self.__train['Text']))
            test_nela = np.array(extract_nela(self.__test['Text']))
            self.__train_input = np.concatenate((train_tfidf, train_nela), axis=1)
            self.__test_input = np.concatenate((test_tfidf, test_nela), axis=1)
        else:
            self.__train_input = train_tfidf
            self.__test_input = test_tfidf

        # Labels
        self.__y_train = self.__train['Category']
        self.__y_test = self.__test['Category']

    def __choose_model(self, mlm_name):
        module_model = ModelSelection.__members__[mlm_name].value
        module_config = ModelConfig.__members__[mlm_name].value

        self.__model = module_model(**module_config)

    def set_model(self, mlm_name):
        self.__choose_model(mlm_name=mlm_name)

    def __fit_model(self):
        self.__model.fit(self.__train_input, self.__y_train)

    def __cv(self):
        # TODO: Cambiar a cross_validate
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.__train_input):
            X_train, X_test = self.__train_input[train_index], self.__train_input[test_index]
            y_train, y_test = self.__y_train[train_index], self.__y_train[test_index]
            self.__model.fit(X_train, y_train)
            y_pred = self.__model.predict(X_test)

            acc = accurcy_score(y_true=X_test, y_pred=y_pred)
            f1 = f1_score(y_true=X_test, y_pred=y_pred, average='macro')
            report = classification_report(y_true=X_test, y_pred=y_pred)
            metrics = {'Accuracy':acc*100, 'F1_macro': f1*100, 'Classification Report': report}

            self.__print_metrics(metrics=metrics)

    def __predict(self):
        y_pred = self.__model.predict(self.__test_input)

        acc = accurcy_score(y_true=self.__y_test, y_pred=y_pred)
        f1 = f1_score(y_true=self.__y_test, y_pred=y_pred, average='macro')
        report = classification_report(y_true=self.__y_test, y_pred=y_pred)
        metrics = {'Accuracy':acc*100, 'F1_macro': f1*100, 'Classification Report': report}

        self.__print_metrics(metrics=metrics)

    def __print_metrics(self, metrics=None):
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for key, value in metrics.items():
            print(str(key) + ': ')
            print(value)
