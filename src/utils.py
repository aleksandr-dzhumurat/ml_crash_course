import logging
import os
import sqlite3

import pandas as pd

import yaml


import nltk
import requests
import pymorphy2

from pathlib import Path
from nltk import sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords

if os.getenv("CONFIG_PATH") is None:
    config_path = "config.yml"
else:
    config_path = os.environ["CONFIG_PATH"]

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class Config:
    def __init__(self, yml_conf):
        self.data_dir = yml_conf["data_dir"]
        self.db_file = os.path.join(yml_conf["data_dir"], yml_conf["sqlite_db_file_name"])
        self.log_file = os.path.join(yml_conf["data_dir"], "service.log")
        self.db_messages_table = "raw_rent_messages"
        self.raw_data_file = os.path.join(yml_conf["data_dir"], "labeled_data_corpus.csv")
        self.model_path = os.path.join(yml_conf["data_dir"], yml_conf["model_file_name"])
        self.tf_idf_params = yml_conf["tf_idf_params"]
        self.vectorizer_path = os.path.join(yml_conf["data_dir"], yml_conf["vectorizer_file_name"])

conf = Config(config)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(conf.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataBase:
    def __init__(self, config):
        self.conn = sqlite3.connect(config.db_file, check_same_thread=False)
        self.conf = config
    
    def run_sql(self, sql_str):
        with self.conn as con:
            res = con.execute(sql_str).fetchall()
        return res

class MessagesDB(DataBase):
    def __init__(self, config):
        super().__init__(config)

    def init_db(self):
        self.run_sql(f"""
                CREATE TABLE IF NOT EXISTS {self.conf.db_messages_table} (
                    msg_id INTEGER NOT NULL,
                    msg TEXT
                );
        """)
        logger.info('table %s created', self.conf.db_messages_table)
        num_rows = self.run_sql(f"""SELECT COUNT(*) as num_cnt FROM {self.conf.db_messages_table}""")[0][0]

        if num_rows == 0:
            msg_df = pd.read_csv(self.conf.raw_data_file)
            try:
                msg_df[['msg_id', 'msg']].to_sql(self.conf.db_messages_table, self.conn, if_exists='replace', index=False)
            except ValueError:
                logger.info('Table already loaded')
            logger.info('data loaded to %s', self.conf.db_messages_table)
        num_rows = self.run_sql(f"""SELECT COUNT(*) as num_cnt FROM {self.conf.db_messages_table}""")[0][0]
        # logger.info(self.run_sql(f"""SELECT sql FROM sqlite_master WHERE name='{self.conf.db_messages_table}';"""))
        logger.info('current rows in table: %d', num_rows)

    
    def read_message(self, msg_id: int):
        msg = {'id': None, 'txt': None}
        sql_str = f"""SELECT msg_id, msg FROM {self.conf.db_messages_table} WHERE msg_id = {msg_id}"""
        msg_raw = self.run_sql(sql_str)
        if len(msg_raw) > 0:
            msg_raw = msg_raw[0]
            msg = {'id': msg_id, 'txt': msg_raw[1]}

        return msg
    
    def get_messages_ids(self):
        res = [int(i[0]) for i in self.run_sql(f"SELECT msg_id FROM {self.conf.db_messages_table} LIMIT 10000")]

        return res

    def get_all_messages(self):
        res = self.run_sql(f"SELECT * FROM {self.conf.db_messages_table} LIMIT 10000")

        return res


nltk.download('punkt')
url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"

def get_text(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')

morph = pymorphy2.MorphAnalyzer()
def normalize_tokens(tokens):
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def remove_stopwords(tokens, stopwords=None, min_length=2):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok
            for tok in tokens
            if tok not in stopwords and len(tok) >= min_length]
    return tokens

stopwords = get_text(url_stopwords_ru).splitlines()
def tokenize_n_lemmatize(text, regexp=r'(?u)\b\w{4,}\b'):
    
    words = [w for sentence in sent_tokenize(text)
            for w in regexp_tokenize(sentence, regexp)]

    words = remove_stopwords(words, stopwords)
    words = normalize_tokens(words)
    return words