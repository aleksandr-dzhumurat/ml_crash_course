"""
----------------------------
Train model
----------------------------
"""

import os
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from utils import conf, logger
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from stacking import my_clf

if __name__ == '__main__':
    input_file = conf.raw_data_file
    if not os.path.exists(input_file):
        raise RuntimeError(f'No input file: {input_file}')
    df = pd.read_csv(input_file)
    train_df = df[df['subset'] == 'train']
    test_df = df[df['subset'] == 'test']
    logger.info('num rows for train: %d', train_df.shape[0])

    X_train = train_df['msg'].values
    y_train = train_df['label']

    X_test = test_df['msg'].values
    y_true = test_df['label']
    # fit 
    vectorizer = TfidfVectorizer(**conf.tf_idf_params).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    lr = LogisticRegression().fit(X_train_csr, y_train)
    # predict
    X_test_csr = vectorizer.transform(X_test)
    y_pred = lr.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)

    logger.info('best_score %.5f', cur_score)

    # ------ YOUR CODE HERE ----------- #
    # train better model
    vectorizer = TfidfVectorizer(max_df=0.2,min_df=12).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    model = my_clf.fit(X_train_csr,y_train) 
    #ExtraTreesClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0).fit(X_train_csr, y_train)

    X_test_csr = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)

    logger.info('Mine best_score %.5f', cur_score)

    # safe better model
    
    model_path = conf.model_path
    pickle.dump([model,vectorizer], open(model_path,'wb'))
    #pickle.dump(vectorizer, open(vectorizer_path,'wb'))
    # --------------------------------- #
