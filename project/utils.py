import gc
import os
import random

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_PATH = 'D:\Projects\ML Bootcamp III\project'

DATA_PATH = os.path.join(PROJECT_PATH, 'data')

ENSEMBLE_RECIPE = {
    '01_XGB_split',
    '02_XGB_flip',
    '03_KNN_flip',

    '04_KNN_POWER',

    '06_NB_flip',
    '07_LR',
    '08_NN_flip',
    '09_NN_extended',
    '10_LGB_extra',

    '11_LGB_split',
    '12_LGB_flip',
    '13_XGB_flip2',
}

eps = 0.0000000001

def logit(X):
    X = X.copy()
    X[X < eps] = eps
    X[(1 - X) < eps] = 1 - eps
    return -np.log((1.0 / X) - 1)


def load_stratify_labels():
    train, _, y = load_data('extended', columns=['clean_start'])
    y[(train['clean_start'] > 0).values] += 2
    return y


def load_data(data_name='default', data_folder='data', columns=None):
    prefix = '' if data_name == 'default' else data_name + '_'
    X_train = pd.read_csv(os.path.join(PROJECT_PATH, data_folder, prefix + 'x_train.csv'), sep=';')
    X_test = pd.read_csv(os.path.join(PROJECT_PATH, data_folder, prefix + 'x_test.csv'), sep=';')
    y_train = pd.read_csv(os.path.join(DATA_PATH, 'y_train.csv'), header=None, sep=';').values.ravel()
    if columns:
        X_train = X_train[columns]
        X_test = X_test[columns]
    return X_train, X_test, y_train


def load_ensemble_data(recipe=None, use_logit=False):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = None

    if not recipe:
        recipe = ENSEMBLE_RECIPE

    for model in recipe:
        xtr, xte, y_train = load_data(data_name=model, data_folder='ensemble')
        X_train = pd.concat((X_train, xtr), axis=1)
        X_test = pd.concat((X_test, xte), axis=1)

    if use_logit:
        for model in recipe:
            xtr, xte, y_train = load_data(data_name=model, data_folder='ensemble')
            xtr_l, xte_l = logit(xtr), logit(xte)
            xtr_l.columns += '_logit'
            xte_l.columns += '_logit'
            X_train = pd.concat((X_train, xtr_l), axis=1)
            X_test = pd.concat((X_test, xte_l), axis=1)
    return X_train, X_test, y_train


def save_result(predictions, model_name, data_folder, test=True, columns=None):
    if columns is None:
        columns = [model_name + '_col_' + str(i + 1) for i in range(predictions.shape[1])] if len(
            predictions.shape) > 1 else [model_name]
    data = pd.DataFrame(predictions, columns=columns)
    filename = model_name + ('_x_test' if test else '_x_train') + '.csv'
    data.to_csv(os.path.join(PROJECT_PATH, data_folder, filename), index=False, sep=';')


def perform_cv(X_train, y_train, clf, model_name, fit_params=None, stratify_labels=None):
    kf = StratifiedKFold(n_splits=5, random_state=2707)
    scores = []
    if fit_params is None:
        fit_params = {}

    if stratify_labels is None:
        stratify_labels = y_train
    for train_idx, test_idx in kf.split(X_train, stratify_labels):
        m = clone(clf)
        m.fit(X_train.iloc[train_idx], y_train[train_idx], **fit_params)
        train_result = m.predict_proba(X_train.iloc[test_idx])[:, 1]
        scores.append(log_loss(y_train[test_idx], train_result))

    print('CV:', scores)
    print('Mean log loss:', np.mean(scores))
    print('Model:', model_name)
    print('Std:', np.std(scores))

    clf = None
    gc.collect()


def VJUH(X_train, X_test, y_train, clf, model_name, folder_name, fit_params=None, seed_name=None, runs=1,
         lasagne=False, stratify_labels=None):
    # in case of using lasagne we should accept np arrays and don't use .loc
    # if isinstance(X_train, pd.DataFrame):
    #     X_train = X_train.values
    #
    # if isinstance(X_test, pd.DataFrame):
    #     X_test = X_test.values

    kf = StratifiedKFold(n_splits=10, random_state=2707, shuffle=True)
    train_result = np.zeros(y_train.shape)
    scores = []
    if fit_params is None:
        fit_params = {}

    fold = 0
    if stratify_labels is None:
        stratify_labels = y_train
    for train_idx, test_idx in kf.split(X_train, stratify_labels):
        fold += 1
        m = clone(clf)
        m.fit(X_train.iloc[train_idx], y_train[train_idx], **fit_params)
        train_result[test_idx] = m.predict_proba(X_train.iloc[test_idx])[:, 1]

        if (seed_name or lasagne) and runs > 1:  # Extra runs
            for i in range(1, runs):
                print(i, 'extra run...')
                m = clone(clf)
                if lasagne:
                    np.random.seed(2707 + i)
                    random.seed(2707 + i)
                else:
                    seed = clf.get_params()[seed_name] + i
                    np.random.seed(seed)
                    random.seed(seed)
                    m.set_params(**{seed_name: seed})
                m.fit(X_train.iloc[train_idx], y_train[train_idx], **fit_params)
                train_result[test_idx] += m.predict_proba(X_train.iloc[test_idx])[:, 1]
            train_result[test_idx] /= runs
        scores.append(log_loss(y_train[test_idx], train_result[test_idx]))
        print('Fold', fold, 'finished with score', scores[-1])

    print(folder_name, 'CV:', scores)
    print('Mean log loss:', np.mean(scores))
    print('Model:', model_name)
    print('Std:', np.std(scores))

    m = clone(clf)
    m.fit(X_train, y_train, **fit_params)
    test_result = m.predict_proba(X_test)[:, 1]

    print('Test calculated')
    if (seed_name or lasagne) and runs > 1:  # Extra runs
        for i in range(1, runs):
            print(i, 'extra run...')
            m = clone(clf)
            if lasagne:
                # nn_rnd.set_rng(2707 + i)
                np.random.seed(2707 + i)
                random.seed(2707 + i)
            else:
                seed = clf.get_params()[seed_name] + i
                np.random.seed(seed)
                random.seed(seed)
                m.set_params(**{seed_name: seed})
            m.fit(X_train, y_train, **fit_params)
            test_result += m.predict_proba(X_test)[:, 1]
        test_result /= runs

    m = None
    gc.collect()

    save_result(train_result, model_name=model_name, data_folder=folder_name, test=False)
    save_result(test_result, model_name=model_name, data_folder=folder_name, test=True)
    print('Results saved!')
