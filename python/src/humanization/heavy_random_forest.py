import logging

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from dataset import read_dataset

logger = logging.getLogger("Heavy chains random forest")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def build_tree(v_type=1) -> CatBoostClassifier:
    dataset = read_dataset(requested_columns=['sequence_alignment_aa', 'v_call'])
    logger.info(f"Tree will train on dataset {dataset.size}")
    X = dataset['sequence_alignment_aa']
    X = X.str.split('', expand=True).fillna(value='')  # Make column for every aa
    y = np.where(dataset['v_call'].apply(lambda x: x.startswith(f"IGHV{v_type}")), 1, 0)
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.2, shuffle=False)

    model = CatBoostClassifier(iterations=200,
                               depth=3,
                               learning_rate=1,
                               loss_function='Logloss',
                               verbose=True)
    model.fit(X_train, y_train,
              cat_features=X.columns.tolist(),
              eval_set=(X_val, y_val))

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    logger.info(f"Test data: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    # Test data: TP=3289, TN=13729, FP=5, FN=1

    return model


if __name__ == '__main__':
    model = build_tree()

