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
import texthero as hero
from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier,AdaBoostClassifier
from sklearn.svm import SVC

if __name__ == '__main__':
    input_file = conf.cleared_data_file
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
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=11).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
   
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=20,
            max_depth=None,
            min_samples_split=2, 
            random_state=0)
            ),
        ('SVC', SVC(random_state=0))
    ]

    classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=BaggingClassifier(LogisticRegression(random_state=0),
        n_estimators=20,
        random_state=0
        )
    )
    
    model = classifier.fit(X_train_csr, y_train)
    X_test_csr = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_csr)
    my_score = f1_score(y_true, y_pred)

    logger.info('My score %.5f', my_score)
    logger.info('Delta %.5f', my_score-cur_score)
    # safe better model

    model_path = conf.model_path
    
    pickle.dump([model,vectorizer], open(model_path,'wb'))
    # --------------------------------- #
