import os
import numpy as np
import pandas as pd
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

from database import Database

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
    RF = {'n_estimators': 100, 'criterion':'gini', 'max_depth': 4} # , 'class_weight': 'balanced'}
    XGB = {'random_state': 42, 'seed':np.random.randint(0, 99999999), 'max_depth':5}
    DT = {'max_depth':50, 'criterion':'gini'} # , 'class_weight': 'balanced'}

class MlModel:
    def __init__(self, train_path, dev_path=None, mlm_name='SVM', language='es', use_ngrams=False, ngrams=1,
                 use_nela_features = False, char_ngram = False):
        self.__train_path = train_path
        self.__dev_path = dev_path
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
        print('Predicting on dev data')
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
        self.__train, self.__dev = load_data_by_extension(self.__train_path, self.__dev_path)
        if self.__dev is None:
           self.__train, self.__dev = split_train_test(self.__train, test_size=0.2)
        else:
            pass
            """print('Tamaño de train: ' + str(len(self.__train)))
            print('Tamaño de dev: ' + str(len(self.__dev)))
            self.__train = pd.concat([self.__train, self.__dev])
            self.__train, self.__dev = split_train_test(self.__train, test_size=0.1)
            print('Tamaño de train: ' + str(len(self.__train)))
            print('Tamaño de dev: ' + str(len(self.__dev)))"""

    def __prepare_input_data(self):
        # Tokenize text
        train_text = tokenize(self.__train['Text'], language=self.__language)
        dev_text = tokenize(self.__dev['Text'], language=self.__language)

        # Delete stopwords
        train_text = remove_stopwords(train_text, language='es')
        dev_text = remove_stopwords(dev_text, language='es')

        # Stem text
        train_text, _ = stemming(train_text)
        dev_text, _ = stemming(dev_text)

        # Tf-Idf
        if self.__char_ngram == 'char':
            train_tfidf, dev_tfidf, vectorizer = tfidf(train_text=self.__train['Text'].to_list(), dev_text=self.__dev['Text'].to_list(),
                                        use_ngrams=self.__use_ngrams, n_grams=self.__ngrams,
                                        type_n_grams=self.__char_ngram)
        else:
            train_tfidf, dev_tfidf, vectorizer = tfidf(train_text=train_text, dev_text=dev_text,
                                        use_ngrams=self.__use_ngrams, n_grams=self.__ngrams,
                                        type_n_grams=self.__char_ngram)
        self.__vectorizer = vectorizer
        # Nela Features
        if self.__use_nela_features:
            train_nela = np.array(extract_nela(self.__train['Text']))
            dev_nela = np.array(extract_nela(self.__dev['Text']))
            self.__train_input = np.concatenate((train_tfidf, train_nela), axis=1)
            self.__dev_input = np.concatenate((dev_tfidf, dev_nela), axis=1)
        else:
            self.__train_input = train_tfidf
            self.__dev_input = dev_tfidf

        # Labels
        self.__y_train = self.__train['Category']
        self.__y_dev = self.__dev['Category']

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
        y_pred = self.__model.predict(self.__dev_input)

        acc = accurcy_score(y_true=self.__y_dev, y_pred=y_pred)
        f1_macro = f1_score(y_true=self.__y_dev, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=self.__y_dev, y_pred=y_pred, average='binary', pos_label='Fake')
        report = classification_report(y_true=self.__y_dev, y_pred=y_pred)
        metrics = {'Accuracy':acc*100, 'F1_macro': f1_macro*100, 'Classification Report': report, 'F1': f1}

        self.__print_metrics(metrics=metrics)

    def __print_metrics(self, metrics=None):
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for key, value in metrics.items():
            print(str(key) + ': ')
            print(value)

    def predict_test_data(self, path_test: str):
        test, _ = load_data_by_extension(path_test)
        test_text = tokenize(test['TEXT'], language=self.__language)
        test_text = remove_stopwords(test_text, language=self.__language)
        test_text, _ = stemming(test_text)
        test_text_tfidf = self.__vectorizer.transform(test_text)
        test_text_tfidf = test_text_tfidf.toarray()
        if self.__use_nela_features:
            test_text_nela = np.array(extract_nela(test['TEXT']))
            test_input = np.concatenate((test_text_tfidf, test_text_nela), axis=1)
        else:
            test_input = test_text_tfidf
        predictions = self.__model.predict(test_input)
        classes = {0:'Fake', 1:'True'}
        # METRICS
        y_true = [classes[p] for p in test['labels']]
        y_pred = predictions 
        acc = accurcy_score(y_true=y_true, y_pred=y_pred)
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label='Fake')
        report = classification_report(y_true=y_true, y_pred=y_pred)
        metrics = {'Accuracy':acc*100, 'F1_macro': f1_macro*100, 'Classification Report': report, 'F1': f1}
        self.__print_metrics(metrics=metrics)
        # SAVING FILE
        ids = test['ID']
        task_name = pd.Series(['fakenews' for _ in range(len(ids))])
        data_file = pd.DataFrame(columns=['TaskName', 'IdentifierOfAnInstance', 'Class'])
        data_file['TaskName'] = task_name
        data_file['IdentifierOfAnInstance'] = ids
        data_file['Class'] = predictions
        filename = 'submissions/submission_with_' + self.__mlm_name + '.txt'
        # filename = 'submission_with_' + self.__mlm_name + '.tsv'
        # data_file.to_csv(filename, sep='\t', index=False, header=None)
        with open(filename, 'w', encoding='utf-8') as fp:
            # fp.write('"TaskName"\t"IdentifierOfAnInstance"\t"Class"')
            for i, row in  data_file.iterrows():
                line = '"' + row['TaskName'] + '" "' + str(row['IdentifierOfAnInstance']) + '" "' + row['Class'] + '"\n"' 
                fp.write(line)