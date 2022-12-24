import logging
import os
import sqlite3

import git.repo.base
import pandas as pd
import yaml


class Config:
    def __init__(self):
        root = self.__get_root()
        yml_conf = self.__load_yml_config(self.__get_config_path(root))
        self.data_dir = os.path.join(root, yml_conf["data_dir"])
        self.bert_dir = os.path.join(self.data_dir, yml_conf["bert_dir"])
        self.db_file = os.path.join(self.data_dir, yml_conf["sqlite_db_file_name"])
        self.log_file = os.path.join(self.data_dir, "service.log")
        self.db_messages_table = "raw_rent_messages"
        self.raw_data_file = os.path.join(self.data_dir, "labeled_data_corpus.csv")
        self.model_path = os.path.join(self.data_dir, yml_conf["model_file_name"])
        self.tf_idf_params = yml_conf["tf_idf_params"]

    @staticmethod
    def __get_root():
        if os.getenv("ROOT") is None:
            root = str(
                git.repo.base.Repo(".", search_parent_directories=True).working_tree_dir
            )
            if root is None:
                raise ValueError("Could not find git repo root")
        else:
            root = os.environ["ROOT"]
        return root

    @staticmethod
    def __get_config_path(root: str):
        if os.getenv("CONFIG_PATH") is None:
            config_path = os.path.join(root, "src/config.yml")
        else:
            config_path = os.environ["CONFIG_PATH"]
        return config_path

    @staticmethod
    def __load_yml_config(config_path: str):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config


conf = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(conf.log_file), logging.StreamHandler()],
    force=True,
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
        self.run_sql(
            f"""
                CREATE TABLE IF NOT EXISTS {self.conf.db_messages_table} (
                    msg_id INTEGER NOT NULL,
                    msg TEXT
                );
        """
        )
        logger.info("table %s created", self.conf.db_messages_table)
        num_rows = self.run_sql(
            f"""SELECT COUNT(*) as num_cnt FROM {self.conf.db_messages_table}"""
        )[0][0]

        if num_rows == 0:
            msg_df = pd.read_csv(self.conf.raw_data_file)
            try:
                msg_df[["msg_id", "msg"]].to_sql(
                    self.conf.db_messages_table,
                    self.conn,
                    if_exists="replace",
                    index=False,
                )
            except ValueError:
                logger.info("Table already loaded")
            logger.info("data loaded to %s", self.conf.db_messages_table)
        num_rows = self.run_sql(
            f"""SELECT COUNT(*) as num_cnt FROM {self.conf.db_messages_table}"""
        )[0][0]
        # logger.info(self.run_sql(f"""SELECT sql FROM sqlite_master WHERE name='{self.conf.db_messages_table}';"""))
        logger.info("current rows in table: %d", num_rows)

    def read_message(self, msg_id: int):
        msg = {"id": None, "txt": None}
        sql_str = f"""SELECT msg_id, msg FROM {self.conf.db_messages_table} WHERE msg_id = {msg_id}"""
        msg_raw = self.run_sql(sql_str)
        if len(msg_raw) > 0:
            msg_raw = msg_raw[0]
            msg = {"id": msg_id, "txt": msg_raw[1]}

        return msg

    def get_messages_ids(self):
        res = [
            int(i[0])
            for i in self.run_sql(
                f"SELECT msg_id FROM {self.conf.db_messages_table} LIMIT 10000"
            )
        ]

        return res
