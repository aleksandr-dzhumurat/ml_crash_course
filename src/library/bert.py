import os
import random
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.nn
import torch.utils.data
from sklearn.metrics import f1_score
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)

from utils import conf, logger


def _get_spits(df: pd.DataFrame):
    Data = namedtuple(
        "Data", ["train_text", "train_labels", "test_text", "test_labels"]
    )

    train_df = df[df.subset == "train"]
    test_df = df[df.subset == "test"]
    logger.info("train_df.shape=%s, test_df.shape=%s", train_df.shape, test_df.shape)
    return Data(
        train_text=train_df.cleared_msg.astype("str").reset_index(drop=True),
        train_labels=train_df.label.reset_index(drop=True),
        test_text=test_df.cleared_msg.astype("str").reset_index(drop=True),
        test_labels=test_df.label.reset_index(drop=True),
    )


def _seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _load_model(model_name: str):
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to("cuda")
    return model


def _load_tokenizer(model_name: str):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer


def _get_tokens(tokenizer: BertTokenizer, texts: List[str], max_seq_len):
    return tokenizer.batch_encode_plus(
        texts, max_length=max_seq_len, padding="max_length", truncation=True
    )


class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {"F1": f1}


def _get_prediction(trainer: Trainer, test_dataset: Data):
    test_pred = trainer.predict(test_dataset)
    labels = np.argmax(test_pred.predictions, axis=-1)
    return labels


def _save_model(model, tokenizer):
    model_path = conf.bert_dir + "/fine-tune-bert"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def train(df: pd.DataFrame):
    data = _get_spits(df)

    hub_model_name = "DeepPavlov/rubert-base-cased-sentence"
    model = _load_model(hub_model_name)
    tokenizer = _load_tokenizer(hub_model_name)

    max_seq_len = df.cleared_msg_len.max()
    tokens_train = _get_tokens(tokenizer, data.train_text.values, max_seq_len)
    tokens_test = _get_tokens(tokenizer, data.test_text.values, max_seq_len)

    _seed_all(42)
    train_dataset = Data(tokens_train, data.train_labels)
    test_dataset = Data(tokens_test, data.test_labels)

    training_args = TrainingArguments(
        output_dir=conf.bert_dir + "/results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_dir=conf.bert_dir + "/logs",
        load_best_model_at_end=True,
        learning_rate=1e-5,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=21,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    _save_model(model, tokenizer)

    pred = _get_prediction(trainer, test_dataset)

    logger.info("f1_score: %s", f1_score(data.test_labels, pred))
