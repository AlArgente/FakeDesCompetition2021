import time
import argparse

from mlmodel import MlModel
from hugginmodel import HugginFaceModel

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
        print('Mode 2: Hugginface Model (bert, distilbert, albert, etc.)')
    elif mode == 1:
        print('ML pipeline:')
        ml = MlModel(train_path='train.xlsx', mlm_name=ml_model, language='es', use_ngrams=False, ngrams=2,
                     use_nela_features=False, test_path='development.xlsx', char_ngram=True)
        ml.pipeline()
    elif mode == 2:
        print('ML pipeline with Cross Validation:')
        ml = MlModel(train_path='train.xlsx', mlm_name=ml_model, language='es', use_ngrams=False, ngrams=2,
                     use_nela_features=False, test_path='development.xlsx', char_ngram=False)
        ml.pipeline2()
    elif mode == 3:
        print('Hugginface pipeline:')
        hugginface = HugginFaceModel(train_path='train.csv', name='distilbert', test_path=None, use_trainer=False)
        hugginface.pipeline()
    else:
        print('Mode is not available now.')

    elapsed_time = time.time() - start_time
    print('The execution took: ' + str(elapsed_time / 60) + ' minutes.')
    print('End of execution.')

if __name__ == '__main__':
    main()
