"""
Train model
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import library.bert as bert
import library.cleaner as cleaner
from utils import conf, logger


def _train_baseline(df: pd.DataFrame) -> None:
    train_df = df[df["subset"] == "train"]
    test_df = df[df["subset"] == "test"]
    logger.info("num rows for train: %d", train_df.shape[0])

    X_train = train_df["msg"].values
    y_train = train_df["label"]

    X_test = test_df["msg"].values
    y_true = test_df["label"]
    # fit
    vectorizer = TfidfVectorizer(**conf.tf_idf_params).fit(X_train)
    X_train_csr = vectorizer.transform(X_train)
    lr = LogisticRegression().fit(X_train_csr, y_train)
    # predict
    X_test_csr = vectorizer.transform(X_test)
    y_pred = lr.predict(X_test_csr)
    cur_score = f1_score(y_true, y_pred)

    logger.info("baseline best_score %.5f", cur_score)
    return None


def _get_labeled_data_corpus() -> pd.DataFrame:
    input_file = conf.raw_data_file
    if not os.path.exists(input_file):
        raise RuntimeError(f"No input file: {input_file}")
    df = pd.read_csv(input_file)
    return df


def _clear_df(df: pd.DataFrame) -> pd.DataFrame:
    df["cleared_msg"] = cleaner.clear(df.msg)

    get_msg_len = lambda msg: len(str(msg).split())
    df["msg_len"] = df.msg.apply(get_msg_len)
    df["cleared_msg_len"] = df.cleared_msg.apply(get_msg_len)

    logger.info("msg_len: min=%d, max=%d", df.msg_len.min(), df.msg_len.max())
    logger.info(
        "cleared_msg_len: min=%d, max=%d",
        df.cleared_msg_len.min(),
        df.cleared_msg_len.max(),
    )

    df = df.loc[df["cleared_msg_len"] != 0].reset_index(drop=True)
    logger.info(
        "cleared_msg_len: min=%d, max=%d",
        df.cleared_msg_len.min(),
        df.cleared_msg_len.max(),
    )
    return df


if __name__ == "__main__":
    df = _get_labeled_data_corpus()
    _train_baseline(df)

    df = _clear_df(df)
    _train_baseline(df)
    bert.train(df)
