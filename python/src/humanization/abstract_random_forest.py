import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.utils import gen_batches

from humanization import config_loader
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Abstract chain RF")


def build_tree(X_train, y_train, val_pool, iterative_learning: bool = False) -> CatBoostClassifier:
    if iterative_learning:
        # Cast to list for length calculation
        batches = list(gen_batches(X_train.shape[0], config.get(config_loader.LEARNING_BATCH_SIZE)))
    else:
        batches = [slice(X_train.shape[0])]
    cnt_batches = len(batches)
    batch_estimators = config.get(config_loader.TOTAL_ESTIMATORS) // cnt_batches
    final_model = None
    for idx, batch in enumerate(batches):
        logger.debug(f"Model training. Batch {idx + 1} of {cnt_batches}")
        X_train_batch = X_train[batch]
        y_train_batch = y_train[batch]

        count_unique = len(np.unique(y_train_batch))
        if count_unique <= 1:
            logger.info("Skip batch with bad diverse target values (at most 1 unique value)")
            continue  # Otherwise, catboost will fail

        train_pool = Pool(X_train_batch, y_train_batch, cat_features=X_train_batch.columns.tolist())

        model = CatBoostClassifier(
            depth=4, loss_function='Logloss', used_ram_limit=config.get(config_loader.MEMORY_LIMIT),
            learning_rate=0.05, verbose=config.get(config_loader.VERBOSE_FREQUENCY),
            max_ctr_complexity=2, n_estimators=batch_estimators)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=25, init_model=final_model)
        final_model = model

        del train_pool

    return final_model
