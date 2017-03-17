import lightgbm as lgb
import numpy as np
import xgboost as xgb

import utils

MODEL_NAME = '13_XGB_flip2'

MODE = 'ensemble'  # 'cv'
CLASSIFIER = 'xgb'  # lgb | xgb
ENSEMBLE_RUNS = 30

PARAMS = {
    'colsample_bytree': 0.75,
    'gamma': 0.0,
    'learning_rate': 0.01,
    'max_depth': 5,
    'min_child_weight': 5,
    'n_estimators': 725,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'subsample': 0.82,

    'nthread': 1,
    'seed': 2707,
    'silent': True
}

COLUMNS = [
    'numberOfDaysActuallyPlayed', 'f__k_4', 'totalScore2', 'f_10',
    'f_old_k_11', 'numberOfAttemptedLevels', 'f_17', 'f_top_to_all2',
    'f__d_6', 'upd_start_level', 'totalBonusScore', 'f_38',
    'averageNumOfTurnsPerCompletedLevel', 'f_old_k_3', 'f_3',
    'last_lvl_completed', 'totalNumOfAttempts2', 'UsefullBoosters2',
    'doReturnOnLowerLevels'
]

if __name__ == '__main__':
    np.random.seed(2707)

    X_train, X_test, y_train = utils.load_data(data_name='flipped2', columns=COLUMNS)

    clf = None
    metric = 'logloss' if CLASSIFIER == 'xgb' else 'binary_logloss'
    if CLASSIFIER == 'xgb':
        clf = xgb.XGBClassifier(**PARAMS)
    else:
        par = PARAMS.copy()
        par['num_leaves'] = 2 ** par['max_depth']
        del par['gamma']
        del par['max_depth']
        clf = lgb.LGBMClassifier(**par)

    if MODE == 'cv':
        utils.perform_cv(X_train, y_train, clf, MODEL_NAME + '-' + CLASSIFIER,
                         fit_params={'eval_metric': metric},
                         stratify_labels=utils.load_stratify_labels()
                         )
    elif MODE == 'ensemble':
        utils.VJUH(X_train, X_test, y_train, clf, MODEL_NAME, 'ensemble',
                   fit_params={'eval_metric': metric},
                   seed_name='seed',
                   stratify_labels=utils.load_stratify_labels(),
                   runs=ENSEMBLE_RUNS
                   )
    else:
        print('Unsupported mode')
