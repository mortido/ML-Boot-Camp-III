import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.base import clone, BaseEstimator

import utils

MODEL_NAME = '01_XGB_split'

# MODE = 'cv'
MODE = 'ensemble'

CLASSIFIER = 'xgb'  # lgb | xgb

START_LEVEL_THRESHOLD = 1
DIRTY_DATA_WEIGHT = 3
NUM_RUNS = 33

X1_PARAMS = {
    'colsample_bytree': 0.7,
    'gamma': 0.975,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 5,
    'n_estimators': 1029 + 71,

    'reg_alpha': 0,
    'reg_lambda': 1.35,
    'subsample': 0.625,

    'nthread': 2,
    'seed': 2707,
    'silent': True
}

X2_PARAMS = {
    'colsample_bytree': 0.9,
    'gamma': 0.9,
    # 'learning_rate': 0.01,
    'learning_rate': 0.007,
    'max_depth': 4,
    'min_child_weight': 4,
    # 'n_estimators': 678 + 70,
    'n_estimators': 1043 + 70,
    'reg_alpha': 1,
    'reg_lambda': 2.5,
    'subsample': 0.775,

    'nthread': 2,
    'seed': 2707,
    'silent': True
}

X1_COLUMNS = [
    'maxPlayerLevel', 'attemptsOnTheHighestLevel', 'totalNumOfAttempts',
    'doReturnOnLowerLevels', 'numberOfBoostersUsed',
    'fractionOfUsefullBoosters', 'totalBonusScore', 'totalStarsCount',
    'numberOfDaysActuallyPlayed', 'last_lvl_completed', 'f_10', 'f_16',
    'f_17', 'f_16_inv',
    'f_17_inv',
    'f_top_to_all', 'f_stars_per_attemp',
    'att_rate', 'f_38', 'f_34', 'f_6_a', 'f_old_7', 'f_old_h_11',
    'f_old_k_11', 'f_old_k_3', 'f_old_5', 'f_old_37', 'f__i_4', 'f__k_4',
    'f__d_6'
]

X2_COLUMNS = [
    'numberOfDaysActuallyPlayed', '%f_15_inv', 'totalBonusScore', '%f_6_a',
    'upd_totalNumOfAttempts', 'upd_totalStarsCount', 'f_17_inv',
    'totalStarsCount', 'f_3', 'f_16', 'f_15_inv', 'f_16+',
    'upd_numberOfDaysActuallyPlayed', '#f_15', 'f_17', '%f_old_40', 'f_17+',
    'att_rate', 'f__9', '%f_38', 'f_15', '%f__k_4', '%f_old_37',
    'upd_completed_levels', 'f_16_inv'
]


class XGBoostSplit(BaseEstimator):
    def __init__(self,
                 base='xgb',
                 colsample_bytree1=0.8,
                 gamma1=0.0,
                 learning_rate1=0.01,
                 max_depth1=5,
                 min_child_weight1=4,
                 n_estimators1=544,
                 subsample1=0.75,
                 reg_alpha1=0,
                 reg_lambda1=1.95,

                 colsample_bytree2=0.8,
                 gamma2=0.0,
                 learning_rate2=0.01,
                 max_depth2=5,
                 min_child_weight2=4,
                 n_estimators2=544,
                 subsample2=0.75,
                 reg_alpha2=0,
                 reg_lambda2=1.95,

                 x1_columns=None,
                 x2_columns=None,

                 num_runs=1,

                 nthread=4,
                 silent=True,
                 seed=2707):

        self.seed = seed
        self.silent = silent
        self.nthread = nthread
        self.reg_lambda2 = reg_lambda2
        self.reg_alpha2 = reg_alpha2
        self.subsample2 = subsample2
        self.n_estimators2 = n_estimators2
        self.min_child_weight2 = min_child_weight2
        self.max_depth2 = max_depth2
        self.learning_rate2 = learning_rate2
        self.gamma2 = gamma2
        self.colsample_bytree2 = colsample_bytree2

        self.reg_lambda1 = reg_lambda1
        self.reg_alpha1 = reg_alpha1
        self.subsample1 = subsample1
        self.n_estimators1 = n_estimators1
        self.min_child_weight1 = min_child_weight1
        self.max_depth1 = max_depth1
        self.learning_rate1 = learning_rate1
        self.gamma1 = gamma1
        self.colsample_bytree1 = colsample_bytree1
        self.base = base

        self.x1_columns = x1_columns if x1_columns else X1_COLUMNS
        self.x2_columns = x2_columns if x2_columns else X2_COLUMNS

        self.num_runs = num_runs

        self.models1 = None
        self.models2 = None

    def fit(self, X, y, sample_weight=None):
        model1 = None
        model2 = None
        if self.base == 'xgb':
            model1 = xgb.XGBClassifier(
                reg_lambda=self.reg_lambda1,
                reg_alpha=self.reg_alpha1,
                subsample=self.subsample1,
                n_estimators=self.n_estimators1,
                min_child_weight=self.min_child_weight1,
                max_depth=self.max_depth1,
                learning_rate=self.learning_rate1,
                gamma=self.gamma1,
                colsample_bytree=self.colsample_bytree1,
                seed=self.seed,
                silent=self.silent,
                nthread=self.nthread

            )
            model2 = xgb.XGBClassifier(
                reg_lambda=self.reg_lambda2,
                reg_alpha=self.reg_alpha2,
                subsample=self.subsample2,
                n_estimators=self.n_estimators2,
                min_child_weight=self.min_child_weight2,
                max_depth=self.max_depth2,
                learning_rate=self.learning_rate2,
                gamma=self.gamma2,
                colsample_bytree=self.colsample_bytree2,
                seed=self.seed,
                silent=self.silent,
                nthread=self.nthread
            )
        else:
            model1 = lgb.LGBMClassifier(
                reg_lambda=self.reg_lambda1,
                reg_alpha=self.reg_alpha1,
                subsample=self.subsample1,
                n_estimators=self.n_estimators1,
                min_child_weight=self.min_child_weight1,
                num_leaves=2 ** self.max_depth1,
                learning_rate=self.learning_rate1,
                colsample_bytree=self.colsample_bytree1,
                seed=self.seed,
                silent=self.silent,
                nthread=self.nthread

            )
            model2 = lgb.LGBMClassifier(
                reg_lambda=self.reg_lambda2,
                reg_alpha=self.reg_alpha2,
                subsample=self.subsample2,
                n_estimators=self.n_estimators2,
                min_child_weight=self.min_child_weight2,
                num_leaves=2 ** self.max_depth2,
                learning_rate=self.learning_rate2,
                colsample_bytree=self.colsample_bytree2,
                seed=self.seed,
                silent=self.silent,
                nthread=self.nthread
            )

        X1, X2, idx1, idx2 = self._split_data(X)
        dirty_weight = np.ones(y.shape[0])
        dirty_weight[idx2] = 3
        metric = 'logloss' if self.base == 'xgb' else 'binary_logloss'
        self.models1 = []
        self.models2 = []
        for i in range(self.num_runs):
            print('Fit run', i)
            m1 = clone(model1)
            m1.set_params(seed=self.seed + i)
            m2 = clone(model2)
            m2.set_params(seed=self.seed + i)
            self.models1.append(m1.fit(X1, y, eval_metric=metric))
            self.models2.append(m2.fit(X2, y, eval_metric=metric, sample_weight=dirty_weight))
        return self

    def predict(self, X):
        X1, X2, idx1, idx2 = self._split_data(X)
        preds = self.models1[0].predict(X1)
        preds[idx2] = self.models2[0].predict(X2.loc[idx2])
        return preds

    def predict_proba(self, X):
        X1, X2, idx1, idx2 = self._split_data(X)
        preds = self.models1[0].predict_proba(X1)
        preds[idx2] = self.models2[0].predict_proba(X2.loc[idx2])
        for i in range(1, self.num_runs):
            preds[idx1] += self.models1[i].predict_proba(X1.loc[idx1])
            preds[idx2] += self.models2[i].predict_proba(X2.loc[idx2])
        return preds / self.num_runs

    def _split_data(self, X):
        idx_1 = (X['upd_start_level'] <= START_LEVEL_THRESHOLD).values
        idx_2 = (X['upd_start_level'] > START_LEVEL_THRESHOLD).values
        X1 = X.loc[:, self.x1_columns]
        X2 = X.loc[:, self.x2_columns]
        return X1, X2, idx_1, idx_2

    def get_best_n_estimator(self):
        max1, max2 = 0, 0
        for i in range(self.num_runs):
            if self.models1[i].best_ntree_limit > max1:
                max1 = max(max1, self.models1[i].best_ntree_limit)
            if self.models2[i].best_ntree_limit < max2:
                max2 = max(max2, self.models2[i].best_ntree_limit)
        return max1, max2


if __name__ == '__main__':
    np.random.seed(2707)

    X_train, X_test, y_train = utils.load_data(data_name='extended')

    clf = XGBoostSplit(num_runs=NUM_RUNS, base=CLASSIFIER)
    if MODE == 'cv':
        utils.perform_cv(X_train, y_train, clf, MODEL_NAME + '-' + CLASSIFIER,
                         stratify_labels=utils.load_stratify_labels())
    elif MODE == 'ensemble':
        utils.VJUH(X_train, X_test, y_train, clf, MODEL_NAME, 'ensemble', stratify_labels=utils.load_stratify_labels())
    else:
        print('Unsupported# mode')
