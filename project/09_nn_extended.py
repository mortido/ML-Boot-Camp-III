import numpy as np
import pandas as pd
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import TrainSplit, BatchIterator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import utils

MODEL_NAME = '09_NN_extended'

# MODE = 'cv'
MODE = 'ensemble'

RUNS = 30

LAYERS = [('input', InputLayer),
          # ('dropout_in', DropoutLayer),

          ('dense1', DenseLayer),
          ('dropout1', DropoutLayer),

          ('dense2', DenseLayer),
          ('dropout2', DropoutLayer),

          ('output', DenseLayer)]

PARAMS = dict(
    layers=LAYERS,
    # dropout_in_p=0,

    dense1_num_units=128,
    dropout1_p=0.01,

    dense2_num_units=256,
    dropout2_p=0.85,
    # dense2_nonlinearity=sigmoid,

    batch_iterator_train=BatchIterator(batch_size=128),
    batch_iterator_test=BatchIterator(batch_size=128),

    output_nonlinearity=softmax,

    update=nesterov_momentum,

    update_learning_rate=0.001,
    update_momentum=0.9,

    train_split=TrainSplit(eval_size=0.2),
    verbose=0,
    max_epochs=160
)

if __name__ == '__main__':
    np.random.seed(2707)

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    X_train, X_test, y_train = utils.load_data('extended')

    X_train = np.log(1 + X_train)
    X_test = np.log(1 + X_test)

    scaler = StandardScaler()
    scaler.fit(pd.concat((X_train, X_test), axis=0, ignore_index=True))
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train).astype(np.int32)
    num_classes = len(encoder.classes_)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    num_features = X_train.shape[1]

    clf = NeuralNet(input_shape=(None, num_features), output_num_units=num_classes, **PARAMS)

    if MODE == 'cv':
        utils.perform_cv(X_train, y_train, clf, MODEL_NAME, stratify_labels=utils.load_stratify_labels())
    elif MODE == 'ensemble':
        utils.VJUH(X_train, X_test, y_train, clf, MODEL_NAME, 'ensemble',
                   lasagne=True,
                   runs=RUNS,
                   stratify_labels=utils.load_stratify_labels()
                   )
    else:
        print('Unsupported mode')
