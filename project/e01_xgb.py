import lightgbm as lgb
import numpy as np
import xgboost as xgb

import utils

MODEL_NAME = 'e01_XGB'
# blend CV: [0.38688803322329518, 0.37359661195200833, 0.36862901648197577, 0.38869388076220612, 0.36886707678778202, 0.39958348470320465, 0.38595356447219353, 0.38423691264850879, 0.36793503273130956, 0.3596784537577068]
# Mean log loss: 0.378406206752
# Model: e01_XGB
# Std: 0.0117784096033

# MODE = 'cv'
MODE = 'out'

CLASSIFIER = 'xgb'  # lgb | xgb

RUNS = 10

PARAMS = {

'colsample_bytree': 0.8,
    'gamma': 0.05,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 2,
    'n_estimators': 700,
    'reg_alpha': 0,
    'reg_lambda': 6.0,
    'subsample': 0.8,

    'nthread': 4,
    'seed': 27,
    'silent': True
}

COLUMNS = [

    '04_KNN_POWER_col_2', '04_KNN_POWER_col_3', '04_KNN_POWER_col_5',
    '04_KNN_POWER_col_6', '01_XGB_split', '08_NN_flip', '11_LGB_split',
    '02_XGB_flip', 'doReturnOnLowerLevels', 'last_lvl_completed',
    'clean_start',

    'totalScore2', 'totalNumOfAttempts',
    'numberOfBoostersUsed', 'f_16_inv',
    'averageNumOfTurnsPerCompletedLevel', 'f_old_7',
    'totalBonusScore2', 'totalStarsCount2'
]

if __name__ == '__main__':
    np.random.seed(2707)

    import pandas as pd

    X_train, X_test, y_train = utils.load_ensemble_data()
    X1, X2, _ = utils.load_data('flipped2')
    X_train = pd.concat((X_train, X1), axis=1)[COLUMNS]
    X_test = pd.concat((X_test, X2), axis=1)[COLUMNS]

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
    elif MODE == 'out':
        utils.VJUH(X_train, X_test, y_train, clf, MODEL_NAME, 'out',
                   fit_params={'eval_metric': metric},
                   seed_name='seed',
                   runs=RUNS,
                   stratify_labels=utils.load_stratify_labels()
                   )
    else:
        print('Unsupported mode')
