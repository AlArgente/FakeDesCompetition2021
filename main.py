import time
import argparse

from mlmodel import MlModel
from hugginmodel import HugginFaceModel
from localattentionmodel import LocalAttentionModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='Select mode of execution', default=1)
    parser.add_argument('--ml_model', type=str, help='ML model to use.', default=None)
    args = vars(parser.parse_args())  # Convert the arguments to a dict
    mode = args['mode']
    ml_model = args['ml_model'].upper() if args['ml_model'] is not None else 'None'
    print('Begin of execution.')
    start_time = time.time()

    if mode == 0:
        print('Modes of execution:')
        print('Mode 0: Help info about the different modes.')
        print('Mode 1: Traditional Machine Learning Model (SVM, RF, Bayes and XGB supported).')
        print('Mode 3: Traditional ML Models to Propaganda Dataset.')
        print('Mode 4: Hugginface Model (bert, distilbert, albert, etc.)')
    elif mode == 1:
        print('ML pipeline:')
        ml = MlModel(train_path='train.xlsx', mlm_name=ml_model, language='es', use_ngrams=True, ngrams=2,
                     use_nela_features=False, dev_path='development.xlsx', char_ngram=False)
        ml.pipeline()
        print('Predicting test data.')
        ml.predict_test_data('test_con_sols.xlsx')
    elif mode == 2:
        print('ML pipeline with Cross Validation:')
        ml = MlModel(train_path='train.xlsx', mlm_name=ml_model, language='es', use_ngrams=False, ngrams=2,
                     use_nela_features=False, dev_path='development.xlsx', char_ngram=False)
        ml.pipeline2()
    elif mode == 3:
        print('ML pipeline with Propaganda dataset:')
        ml = MlModel(train_path='./propagandadata/train_binary.tsv', mlm_name=ml_model, language='es', use_ngrams=False, ngrams=2,
                     use_nela_features=True, dev_path='./propagandadata/test_binary.tsv', char_ngram=False)
        ml.pipeline()
    elif mode == 4:
        print('Hugginface pipeline:')
        hugginface = HugginFaceModel(train_path='train.xlsx', name='distilbert', test_path='development.xlsx', use_trainer=False)
        hugginface.pipeline()
    elif mode == 5:
        print('Local-Attention Model:')
        attmodel = LocalAttentionModel(batch_size=32, epochs=250, optimizer='adam', max_sequence_len=420, lstm_units=32, 
                                    path_train='train.xlsx', path_dev='development.xlsx', learning_rate=1e-4, dense_units=32,
                                    length_type='fixed', dr_rate=0.2)
        
        attmodel.pipeline()
        attmodel.fit()
        print('Predicting on dev data')
        attmodel.predict()
        print('Predicting test data.')
        attmodel.predict_test_data('test.xlsx')
    else:
        print('Mode is not available now.')

    elapsed_time = time.time() - start_time
    print('The execution took: ' + str(elapsed_time / 60) + ' minutes.')
    print('End of execution.')

if __name__ == '__main__':
    main()
