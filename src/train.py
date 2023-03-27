import os
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score

from utils import conf, logger
import pickle

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
    # ------ YOUR CODE HERE ----------- #
    # train better model
    
    gbr = GradientBoostingRegressor(random_state=1, n_estimators=500, max_depth=8).fit(X_train_csr, y_train)
    # predict
    X_test_csr = vectorizer.transform(X_test)
    y_pred = gbr.predict(X_test_csr)
    y_pred = [0 if i < 0.5 else 1 for i in y_pred]
    cur_score = f1_score(y_true, y_pred)

    logger.info('best_score %.5f', cur_score)

    # safe better model
    
    pkl_filename = conf.vector_path
    with open(pkl_filename, 'wb') as file: 
        pickle.dump(vectorizer, file)
    
    pkl_filename = conf.model_path 
    with open(pkl_filename, 'wb') as file: 
        pickle.dump(gbr, file)

    # --------------------------------- #