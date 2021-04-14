import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFTrainer, TFTrainingArguments, AutoConfig
from utils import *


# HUGGINGACE MODEL WITH TENSORFLOW

class HugginFaceModel:
    def __init__(self, name='distilbert', train_path='train.csv', test_path=None, use_trainer=False):
        self.__model_name = 'distilbert-base-uncased'
        self.__train_path = train_path
        self.__test_path = test_path
        self.name = name

        # For Deep Learning
        self.__batch_size = 16
        self.__max_seq_length = 128
        self.__epochs = 1
        self.__learning_rate = 5e-5
        self.__use_trainer = use_trainer

    def __load_data_csv(self):
        """Function that reads data from csv
        :return: load train, test data to class
        """
        self.__train, self.__test = load_data_xlsx(self.__train_path, self.__test_path)
        if self.__test is None:
            self.__train, self.__test = split_train_test(self.__train)

    def __load_data_as_tensors(self):
        """Function that load data for training_dataset
        :return: train_dataset, test_dataset
        """
        # Load tokenizer
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        # config = AutoConfig.from_pretrained(self.__model_name)
        # Get text
        train_text = self.__train['Text'].to_list()
        test_text = self.__test['Text'].to_list()
        self.__y_train = self.__train['Category']
        self.__y_test = self.__test['Category']

        # Prepare encodings
        train_encodings = self.__tokenizer(train_text, truncation=True, padding=True, max_length=self.__max_seq_length)
        test_encoddings = self.__tokenizer(test_text, truncation=True, padding=True)

        # Prepare tensorflow datasets
        self.__train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.__y_train
        ))

        self.__test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encoddings),
            self.__y_test
        ))

    def __fit(self):
        if self.__use_trainer:
            self.__fit_model_with_trainer()
        else:
            self.__fit_model()

    def __fit_model_with_trainer(self):
        """Function that fit the model using the trainer class
        :return: None
        """
        # Create training args
        self.__training_args = TFTrainingArguments(
            output_dir='./results/',
            num_train_epochs=self.__epochs,
            per_device_train_batch_size=self.__batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs/',
            logging_steps=10,
        )
        # Load model
        with self.__training_args.strategy.scope():
            self.__model = TFAutoModelForSequenceClassification.from_pretrained(self.__model_name)
        # Prepare Trainer
        self.__trainer = TFTrainer(
            model=self.__model,
            args=self.__training_args,
            train_dataset=self.__train_dataset
        )
        # Train
        self.__trainer.train()

    def __predict_trainer(self):
        return self.__trainer.predict(self.__test_dataset)

    def __fit_model(self):
        """Function that fits the model with traditional methods
        :return:
        """

        self.__config = AutoConfig.from_pretrained(self.__model_name,
                                                   num_labels = 2,
                                                   label2id = (0, 1),
                                                   finetuning_tasks = 'text-classification')

        self.__model = TFAutoModelForSequenceClassification.from_pretrained(self.__model_name,
                                                                            config=self.__config)
        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)

        self.__model.compile(optimizer=self.__optimizer, loss=self.__model.compute_loss)

        self.__model.summary()

        self.__model.fit(self.__train_dataset.shuffle(1000).batch(self.__batch_size), epochs=self.__epochs,
                         batch_size=self.__batch_size)

    def __predict_model(self):
        return self.__model.predict(self.__test_dataset)

    def __predict(self):
        # self.__test_dataset=self.__test_dataset[0]
        # self.__y_test = self.__y_test[0]
        if self.__use_trainer:
            pred = self.__predict_trainer()
        else:
            pred = self.__predict_model()

        print(pred)
        output = pred['logits']
        print(output)
        y_preds = tf.nn.softmax(output, axis=1).numpy()
        # y_pred = []
        print(y_preds)
        print('Calculo y_pred')
        y_pred = np.argmax(y_preds, axis=1)
        # y_pred = [np.argmax(p) for p in y_preds]
        # print(y_pred)
        """
        for p in y_preds:
            if p[0] > p[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
        """
        print(len(y_pred))
        assert len(y_pred) == len(self.__y_test)
        metrics = acc_f1macro_creport(y_true=self.__y_test, y_pred=y_pred)

        self.__print_metrics(metrics=metrics)

    def __print_metrics(self, metrics=None):
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        for key, value in metrics.items():
            print(str(key) + ': ')
            print(value)

    def pipeline(self):
        """Pipeline for the class
        :return:
        """
        # Read data
        print('Loading data')
        self.__load_data_csv()

        # Generate datasets
        self.__load_data_as_tensors()

        # Fit the model
        self.__fit()

        # Predict
        self.__predict()
