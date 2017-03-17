import lightgbm as lgb
import numpy as np
import xgboost as xgb

import utils

MODEL_NAME = '02_XGB_flip'

MODE = 'ensemble'  # 'cv'
CLASSIFIER = 'xgb'  # lgb | xgb
ENSEMBLE_RUNS = 60

PARAMS = {
    'colsample_bytree': 0.725,
    'gamma': 0.2,
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 5,
    'n_estimators': 743,
    'reg_alpha': 0.825,
    'reg_lambda': 1,
    'subsample': 0.55,

    'nthread': 4,
    'seed': 2707,
    'silent': True
}

COLUMNS = [
    'maxPlayerLevel', 'attemptsOnTheHighestLevel', 'doReturnOnLowerLevels',  # 0.380110811167
    'fractionOfUsefullBoosters', 'upd_start_level', 'totalNumOfAttempts',
    'numberOfAttemptedLevels', 'averageNumOfTurnsPerCompletedLevel',
    'numberOfBoostersUsed', 'totalScore', 'totalBonusScore',
    'totalStarsCount', 'numberOfDaysActuallyPlayed', 'f_old_7', 'f_old_5',
    'f_38', 'f_old_37', 'f_old_40', 'f_10', 'f__9', 'f_old_h_11',
    'f_old_k_11', 'f__d_6', 'f_3', 'f_15_inv', 'f_17_inv', 'f__k_4', 'f_6_a',
]

if __name__ == '__main__':
    np.random.seed(2707)

    X_train, X_test, y_train = utils.load_data(data_name='log_flipped', columns=COLUMNS)

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
