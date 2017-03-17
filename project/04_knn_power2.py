import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import utils

MODEL_NAME = '04_KNN_POWER_'

MAX_POWER = 10
CALIBRATION_CV = 10

if __name__ == '__main__':
    np.random.seed(2707)

    X_train, X_test, y_train = utils.load_data(data_name='log_flipped')
    scaler = StandardScaler()
    scaler.fit(pd.concat((X_train, X_test), axis=0, ignore_index=True))
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    kf = StratifiedKFold(n_splits=CALIBRATION_CV, shuffle=True, random_state=4891)
    for i in range(1, MAX_POWER + 1):
        clf = KNeighborsClassifier(n_neighbors=2 ** i, n_jobs=1)
        clf = CalibratedClassifierCV(clf, method='isotonic', cv=kf)
        utils.VJUH(X_train, X_test, y_train, clf, MODEL_NAME + str(i), 'ensemble',
                   stratify_labels=utils.load_stratify_labels())

    train = []
    test = []
    for i in range(1, MAX_POWER + 1):
        tr, ts, _ = utils.load_data(data_name=MODEL_NAME + str(i), data_folder='ensemble')
        train.append(tr)
        test.append(ts)
    result = pd.concat(train, axis=1, ignore_index=True)
    utils.save_result(result.values, MODEL_NAME[:-1], data_folder='ensemble', test=False)
    result = pd.concat(test, axis=1, ignore_index=True)
    utils.save_result(result.values, MODEL_NAME[:-1], data_folder='ensemble', test=True)
