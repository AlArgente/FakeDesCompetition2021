import os
import statistics
from nltk.tokenize import sent_tokenize
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, \
    Dropout, Dense, Concatenate, SpatialDropout1D, Embedding, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns 

from utils import *
from attention_layers import Attention
from factory_embeddings import FactoryEmbeddings

class LocalAttentionModel:

    def __init__(self, batch_size, epochs, optimizer, max_sequence_len, lstm_units,
                 path_train, path_test=None, vocab_size=None, l2_rate=1e-5, path_dev=None,
                 learning_rate=1e-3, dr_rate=0.2, max_sequence_len_headline = 120,
                 embedding_size=300, max_len=10000, load_embeddings=True, buffer_size=3, emb_type='fasttext',
                 length_type='median', dense_units=128, n_class=False):
    
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__max_sequence_len = max_sequence_len
        self.__lstm_units = lstm_units
        self.__learning_rate = learning_rate
        self.__path_train = path_train
        self.__path_test = path_test
        self.__path_dev = path_dev
        self.__l2_rate = l2_rate
        self.__dr_rate = dr_rate
        self.__embedding_size = embedding_size
        self.__max_len = max_len
        self.__emb_type = emb_type
        self.__length_type = length_type
        self.__dense_units = dense_units
        self.__n_class = n_class
        self.__max_sequence_len_headline = max_sequence_len_headline

        self.__model = None
        pos = 5737
        neg = 45557
        total = 51294
        weight_for_0 = (1 / neg) * total / 2.0
        weight_for_1 = (1 / pos) * total / 2.0
        self.__class_weights = {0: weight_for_0, 1: weight_for_1} # Update class weights if necessary.
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        self.OPTIMIZERS = {
            'adam': Adam(learning_rate=self.__learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            'rmsprop': RMSprop(learning_rate == self.__learning_rate)
        }
        self.__optimizer = self.OPTIMIZERS[optimizer]

        self.__METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
        self.__callbacks = None
        self.__nb_words = None
        self.__embedding_matrix = None
        self.__train = None
        self.__test = None
        self.__dev = None
        self.__word_index = None

        # FOR CALLBACKS
        self.checkpoint_filepath = './checkpoints/checkpoint_attention.cpk'
        self.model_save = ModelCheckpoint(filepath=self.checkpoint_filepath, save_weights_only=True, mode='auto',
                                          monitor='loss', save_best_only=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, verbose=1, mode='auto',
                                           min_lr=(5e-5))
        self.early_stop = EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='min',
            baseline=None, restore_best_weights=True
        )
        self.__callbacks = [self.model_save, self.reduce_lr] #, self.early_stop

    @property
    def model(self):
        return self.__model

    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def epochs(self):
        return self.__epochs

    def __print_configuration(self):
        print('Embeddings: ' + self.__emb_type)
        print('Embeddings size: ' + str(self.__embedding_size))
        print('Max_sequence_lenthg: ' + self.__length_type)
        print('Batch_size: ' + str(self.__batch_size))
        print('Epochs: ' + str(self.__epochs))
        print('Learning rate: ' + str(self.__learning_rate))
        print('LSTM units: ' + str(self.__lstm_units))
        print('Dense units: ' + str(self.__dense_units))
        print('Dropout rate: ' + str(self.__dr_rate))
        print('Number of classes: ' + str(self.__n_class))

    def build(self):
        # Headlines input
        headline_input = Input(shape=(self.__max_sequence_len_headline,), dtype="int32", name='headline_input')
        
        embeddings_hi = Embedding(self.__nb_words, self.__embedding_size, 
                                weights=[self.__embedding_matrix], input_length=self.__max_sequence_len_headline,
                                trainable=False, name='Embeddings_headline')
        embeddings_headline = embeddings_hi(headline_input)
        embeddings_headline = SpatialDropout1D(0.2)(embeddings_headline)
        x_headline, forward_h_headline, forward_c_headline, backward_h_headline, backward_c_headline = Bidirectional(LSTM(units=self.__lstm_units, activation='tanh',
                                                                            return_sequences=True, recurrent_dropout=0,
                                                                            recurrent_activation='sigmoid', 
                                                                            unroll=False, use_bias=True, 
                                                                            kernel_regularizer=l1_l2(l2=self.__l2_rate),
                                                                            return_state=True),
                                                                        name='bilstm_headline')(embeddings_headline)
        
        # hidden_states_headline = Concatenate()([forward_h_headline, backward_h_headline])
        """
        attention_input_headline = [x_headline, hidden_states_headline]
        encoder_output_headline, att_weights_headline = Attention(context='many-to-one',
                                                alignment_type='local-p',
                                                window_width=100,
                                                score_function='scaled_dot',
                                                name='local-att-layer_headline'
                                                )(attention_input_headline)
        out_headline = Dense(units=self.__dense_units, activation='tanh', kernel_regularizer=l1_l2(l2=self.__l2_rate), name='dense_headline')(encoder_output_headline)
        out_headline = Dropout(rate=self.__dr_rate, name='dropout_headline')(out_headline)
        out_headline = GlobalMaxPool1D(name='MaxPool_headline')(out_headline)
        """
        # News input
        sequence_input = Input(shape=(self.__max_sequence_len,), dtype="int32", name="seq_input")
        news_inputs = Concatenate()([headline_input, sequence_input])
        embeddings = Embedding(self.__nb_words, self.__embedding_size, 
                                # weights=[self.__embedding_matrix], input_length=self.__max_sequence_len,
                                weights=[self.__embedding_matrix], input_length=(self.__max_sequence_len+self.__max_sequence_len_headline),
                                trainable=False, name='Embeddings')
        
        # embedding_sequence = embeddings(sequence_input)
        embedding_sequence = embeddings(news_inputs)
        embedding_sequence = SpatialDropout1D(0.2)(embedding_sequence)

        x, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(units=self.__lstm_units, activation='tanh',
                                                                            return_sequences=True, recurrent_dropout=0,
                                                                            recurrent_activation='sigmoid', 
                                                                            unroll=False, use_bias=True, 
                                                                            kernel_regularizer=l1_l2(l2=self.__l2_rate),
                                                                            return_state=True),
                                                                        # name='bilstm')(embedding_sequence)
                                                                        name='bilstm')(inputs=embedding_sequence, initial_state=[forward_h_headline, forward_c_headline, backward_h_headline, backward_c_headline])

        # hidden_states = Concatenate()([forward_h, backward_h])
        # attention_input = [x, hidden_states]

        hidden_states_new = Concatenate()([forward_h, backward_h])
        x_new = Concatenate(axis=1)([x_headline, x])
        attention_input_new = [x_new, hidden_states_new]

        encoder_output, att_weights = Attention(context='many-to-one',
                                                alignment_type='local-p',
                                                window_width=100,
                                                score_function='scaled_dot',
                                                name='local-att-layer'
                                                # )(attention_input)
                                                )(attention_input_new)

        out = Dense(units=self.__dense_units*3, activation='tanh', kernel_regularizer=l1_l2(l2=self.__l2_rate), name='dense1')(encoder_output)
        out = Dropout(rate=self.__dr_rate, name='dropout1')(out)
        out = Dense(units=self.__dense_units*2, activation='tanh', kernel_regularizer=l1_l2(l2=self.__l2_rate), name='dense2')(out)
        out = Dropout(rate=self.__dr_rate, name='dropout2')(out)
        out = Dense(units=self.__dense_units, activation='tanh', kernel_regularizer=l1_l2(l2=self.__l2_rate), name='dense3')(out)
        out = Dropout(rate=self.__dr_rate, name='dropout3')(out)
        out = GlobalMaxPool1D(name='MaxPool')(out)

        # Aggregate both outputs
        # o = Concatenate()([out, out_headline])
        # o = Dense(units=self.__dense_units, activation='relu', kernel_regularizer=l1_l2(l2=self.__l2_rate), name='Aggregation')(o)
        # o = Dropout(rate=self.__dr_rate, name='aggregation_dropout')(o)
        # Predictions
        prediction = Dense(units=2, activation='softmax', name='prediction')(out)

        self.__model = Model(inputs=[sequence_input, headline_input], outputs=[prediction], name='local-attention-model')
        self.__model.compile(loss=BinaryCrossentropy(),
                            optimizer=self.__optimizer,
                            metrics=self.__METRICS)

        self.__model.summary()

    def __print_metrics(self, metrics=None):
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for key, value in metrics.items():
            print(str(key) + ': ')
            print(value)
    
    def predict_test_data(self, path_test: str):
        # preds = self.__model.predict()
        test, _ = load_data_by_extension(path_test)
        test_text = test.TEXT
        test_headline = [headline if type(headline) is str else '' for headline in test.HEADLINE.to_list()]
        word_seq_test = self.__tokenizer.texts_to_sequences(test_text)
        word_seq_test_headline = self.__tokenizer.texts_to_sequences(test_headline)
        X_test = pad_sequences(word_seq_test, self.__max_sequence_len)
        X_test_headlines = pad_sequences(word_seq_test_headline, self.__max_sequence_len_headline)
        self.X_test = X_test
        self.X_test_headlines = X_test_headlines
        predictions = self.__model.predict({'seq_input':X_test, 'headline_input':X_test_headlines})
        predictions = [np.argmax(p) for p in predictions]
        classes = {0:'Fake', 1:'True'}
        predictions = [classes[p] for p in predictions]
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
        filename = 'submissions/submission_with_' + 'attention_model' + '.txt'
        # filename = 'submission_with_' + self.__mlm_name + '.tsv'
        # data_file.to_csv(filename, sep='\t', index=False)
        with open(filename, 'w') as fp:
            # fp.write('"TaskName"\t"IdentifierOfAnInstance"\t"Class"\n')
            for i, row in  data_file.iterrows():
                line = '"' + row['TaskName'] + '"\t"' + str(row['IdentifierOfAnInstance']) + '"\t"' + (row['Class']) + '"\n' 
                fp.write(line)

    def pipeline(self):
        self.__print_configuration()
        print('Loading data')
        self.__load_data()
        print('Padding sentences')
        self.__pad_sentences()
        print('Loading embeddings')
        self.__load_embeddings()
        self.__prepare_data_as_tensors()
        print('Building the model')
        self.build()


    def __load_data(self):
        self.__train, self.__dev = load_data_by_extension(self.__path_train, self.__path_dev)
        if self.__dev is None:
            self.__train, self.__dev = split_train_test(self.__train, test_size=0.1)
        else:
            print('Tamaño de train: ' + str(len(self.__train)))
            print('Tamaño de dev: ' + str(len(self.__dev)))
            self.__train = pd.concat([self.__train, self.__dev])
            self.__train, self.__dev = split_train_test(self.__train, test_size=0.1)
            print('Tamaño de train: ' + str(len(self.__train)))
            print('Tamaño de dev: ' + str(len(self.__dev)))

        classes = {'Fake':0, 'True':1}
        
        self.y_train = [classes[c] for c in self.__train.Category]
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=2)
        
        self.y_dev = [classes[c] for c in self.__dev.Category]
        self.y_dev = tf.keras.utils.to_categorical(self.y_dev, num_classes=2)

    def __pad_sentences(self):
        if self.__emb_type == 'glove':
            self.__max_len = 40000
        elif self.__emb_type == 'fasttext':
            self.__max_len = 200000
        else:
            raise ValueError('Not a valid type')

        train_headline = self.__train.Headline
        dev_headline = self.__dev.Headline


        tokenizer = Tokenizer(num_words=None, lower=True, char_level=False, split=' ')

        full_text = pd.concat([self.__train.Text, self.__dev.Text, train_headline, dev_headline])

        tokenizer.fit_on_texts(full_text)
        self.__tokenizer = tokenizer
        self.__word_index = tokenizer.word_index
        word_seq_train = tokenizer.texts_to_sequences(self.__train.Text)
        word_seq_dev = tokenizer.texts_to_sequences(self.__dev.Text)
        word_seq_train_headline = tokenizer.texts_to_sequences(train_headline)
        word_seq_dev_headline = tokenizer.texts_to_sequences(dev_headline)

        if self.__length_type.lower() == 'fixed':
            print('Se usará {} como max_sequence_len.', self.__max_sequence_len)
        elif self.__length_type.lower() == 'mean':
            print('Se usará la media como max_sequence_len.')
            self.__max_sequence_len = self.__mean_padding(word_seq_train)
        elif self.__length_type.lower() == 'mode':
            print('Se usará la moda como max_sequence_len.')
            self.__max_sequence_len = self.__mode_padding(word_seq_train)
        elif self.__length_type.lower() == 'median':
            print('Se usará la mediana como max_sequence_len.')
            self.__max_sequence_len = self.__median_padding(word_seq_train)
        else:
            print('The padding used will be the fixed one.')
        print('The max_sequence_len is: ', self.__max_sequence_len)

        self.X_train = pad_sequences(word_seq_train, maxlen=self.__max_sequence_len)
        self.X_dev = pad_sequences(word_seq_dev, maxlen=self.__max_sequence_len)
        self.X_train_headline = pad_sequences(word_seq_train_headline, maxlen=self.__max_sequence_len_headline)
        self.X_dev_headline = pad_sequences(word_seq_dev_headline, maxlen=self.__max_sequence_len_headline)
    
    def __load_embeddings(self):
        file_embeddings = 'embeddings_fakedes.npy'
        if os.path.exists(file_embeddings):
            self.__embedding_matrix = np.load(file_embeddings)
            self.__nb_words = len(self.__embedding_matrix)
            print('Embeddings cargados de fichero.')
        else:
            self.__emb = FactoryEmbeddings()
            self.__emb.load_embeddings(self.__emb_type)
            nb_words = max(self.__max_len, len(self.__word_index))
            embeddings = self.__emb.embeddings.embeddings_full
            embedding_matrix = np.zeros((nb_words, self.__embedding_size))
            # unknown_emb = np.random.normal(0, 1, 300)
            words_not_found = []
            for word, i in self.__word_index.items():
                if i > self.__nb_words:
                    continue
                embedding_vector = embeddings.get(word)
                if embedding_vector is not None and len(embedding_vector) > 0:
                    embedding_matrix[i] = embedding_vector
                else:
                    words_not_found.append(word)
            np.save(file_embeddings, embedding_matrix)
            print('Embeddings guardados.')
            self.__nb_words = nb_words
            self.__embedding_matrix = embedding_matrix
    
    def __prepare_data_as_tensors(self):
        print('Loading data as tensors')
        print(len(self.X_train))
        print(len(self.X_train_headline))
        self.__train_dataset = tf.data.Dataset.from_tensor_slices(({'seq_input':self.X_train, 'headline_input':self.X_train_headline}, self.y_train))
        self.__train_dataset = self.__train_dataset.shuffle(len(self.X_train)).batch(self.__batch_size)

        self.__val_dataset = tf.data.Dataset.from_tensor_slices(({'seq_input':self.X_dev, 'headline_input':self.X_dev_headline}, self.y_dev))
        self.__val_dataset = self.__val_dataset.shuffle(len(self.X_dev)).batch(self.__batch_size)

    def fit(self):
        tf.random.set_seed(42)
        self.__history = self.__model.fit(self.__train_dataset, epochs=self.__epochs, batch_size=self.__batch_size,
                                        verbose=1, callbacks=self.__callbacks, shuffle=True,
                                        validation_data=self.__val_dataset)

    def predict(self):
        predictions = self.__model.predict(self.X_dev)
        predictions = [np.argmax(p) for p in predictions]
        classes = {0:'Fake', 1:'True'}
        predictions = [classes[p] for p in predictions]
        acc = accurcy_score(y_true=self.__dev.Category, y_pred=predictions)
        f1_macro = f1_score(y_true=self.__dev.Category, y_pred=predictions, average='macro')
        f1 = f1_score(y_true=self.__dev.Category, y_pred=predictions, average='binary', pos_label='Fake')
        report = classification_report(y_true=self.__dev.Category, y_pred=predictions)
        metrics = {'Accuracy':acc*100, 'F1_macro': f1_macro*100, 'Classification Report': report, 'F1': f1}

        self.__print_metrics(metrics=metrics)

    def __print_metrics(self, metrics=None):
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for key, value in metrics.items():
            print(str(key) + ': ')
            print(value)

    def __mean_padding(self, text):
        lst = []
        for sequence in text:
            lst.append(len(sequence))
        return int(statistics.mean(lst))

    def __mode_padding(self, text):
        lst = []
        for sequence in text:
            lst.append(len(sequence))
        return int(statistics.mode(lst))

    def __median_padding(self, text):
        lst = []
        for sequence in text:
            lst.append(len(sequence))
        return int(statistics.median(lst))

    def __sentence_median_padding(self, text):
        """Function that calculate the median of sentences in the text to pad the mean_model to that maximum
        sentences per sequence.
        Args:
            - text: List or pandas series with all the sequence to get the median
        Returns:
            - The padding to apply
        """
        lst = []
        for sequence in text:
            lst.append(len(sent_tokenize(sequence)))
        return int(statistics.median(lst))