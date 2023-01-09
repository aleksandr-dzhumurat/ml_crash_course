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
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score

from utils import conf, logger, tokenize_n_lemmatize

import pickle

from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

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
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize_n_lemmatize).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    X_test_csr = vectorizer.transform(X_test)

    base_models = [('svm', SVC(gamma=0.44, random_state=0)), ('mnb', MultinomialNB(alpha=0.35))]
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(estimators=base_models, 
                                        final_estimator=meta_model, 
                                        passthrough=True, 
                                        cv=2).fit(X_train_csr, y_train)
    y_train_pred = stacking_model.predict(X_train_csr)
    y_pred = stacking_model.predict(X_test_csr)

    logger.info('Confusion Matrix:\n%s', str(confusion_matrix(y_true, y_pred)))

    logger.info('train_f1: {:.5f} f1: {:.5f} accuracy: {:.5f} recall_score:{:.5f} AUC_score: {:.5f}'.format( 
                                f1_score(y_train, y_train_pred),
                                f1_score(y_true, y_pred),
                                accuracy_score(y_true, y_pred),
                                recall_score(y_true, y_pred),
                                roc_auc_score(y_true, y_pred)))

    model_path = conf.model_path

    with open(conf.vectorizer_path, 'wb') as fin:
        pickle.dump(vectorizer, fin)

    with open(model_path, 'wb') as fin:
        pickle.dump(stacking_model, fin)
    # --------------------------------- #
