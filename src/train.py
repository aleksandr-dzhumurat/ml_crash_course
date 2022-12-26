"""
----------------------------
Train model
----------------------------
"""

import os
from typing import Dict

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from utils import conf, logger, preprocess


if __name__ == '__main__':
    input_file = conf.raw_data_file
    if not os.path.exists(input_file):
        raise RuntimeError(f'No input file: {input_file}')
    df = pd.read_csv(input_file)
    train_df = df[df['subset'] == 'train']
    test_df = df[df['subset'] == 'test']
    logger.info('num rows for train: %d', train_df.shape[0])

    X_train = preprocess(train_df['msg'].values)
    y_train = train_df['label']

    X_test = preprocess(test_df['msg'].values)
    y_true = test_df['label']
    # fit
    vectorizer = TfidfVectorizer(**conf.tf_idf_params).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    lr = LogisticRegression().fit(X_train_csr, y_train)
    # predict
    X_test_csr = vectorizer.transform(X_test)
    y_pred = lr.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)

    logger.info('LogisticRegression best_score %.5f', cur_score)

    # ------ YOUR CODE HERE ----------- #
    # train better model

    SEED = 123

    rf = RandomForestClassifier(random_state=SEED).fit(X_train_csr, y_train)
    y_pred = rf.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)
    logger.info('RandomForestClassifier best_score %.5f', cur_score)

    gb = GradientBoostingClassifier(max_depth=10, random_state=SEED).fit(X_train_csr, y_train)
    y_pred = gb.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)
    logger.info('GradientBoostingClassifier best_score %.5f', cur_score)

    # save trained model
    pickle.dump(gb, open(conf.model_path, 'wb'))
    pickle.dump(vectorizer, open(conf.vec_model_path, 'wb'))

    # --------------------------------- #
