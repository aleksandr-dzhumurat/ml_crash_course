"""
Web-service with API
"""
import os
import unicodedata

import numpy as np
from flask import Flask, render_template, request
from flask_restful import Api, Resource, reqparse
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# from library.cleaner import clear
from src.utils import MessagesDB, conf, logger

db = MessagesDB(conf)
db.init_db()


def load_pipe(model_path):
    # load trained model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    logger.info("tokenizer loaded from %s", model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    logger.info("model loaded from %s", model_path)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe


app = Flask(__name__)
api = Api(app)
pipe = load_pipe(conf.hub_fine_tune_model)


@app.route("/messages/<string:identifier>")
def pridict_labell(identifier):
    msg = db.read_message(msg_id=int(identifier))
    # model predict single label
    if not msg:
        logger.info("message %s not found", identifier)
        return render_template("index.html", txt="message not found")
    else:
        pred = pipe(msg["txt"])
        if not pred:
            logger.info("can't predict label for msg %s", identifier)
            return render_template("index.html", txt="can't predict label")

        predicted_label = pred
        return render_template(
            "page.html", id=identifier, txt=msg["txt"], label=predicted_label
        )


@app.route("/feed/")
def feed():
    limit = request.args.get("limit", 10)
    limit = int(limit)
    # rank all messages and predict

    msgs = db.get_messages_ids(limit)

    recs = [
        {"msg_id": 1, "msg_txt": "example_txt_1"},
        {"msg_id": 2, "msg_txt": "example_txt_2"},
    ]
    return render_template("feed.html", recs=recs)


class Messages(Resource):
    def __init__(self):
        super(Messages, self).__init__()
        self.msg_ids = db.get_messages_ids()  # type: list

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("limit", type=int, default=10, location="args")
        args = parser.parse_args()
        try:
            resp = [
                int(i)
                for i in np.random.choice(self.msg_ids, size=args.limit, replace=False)
            ]
        except ValueError as e:
            resp = "Error: Cannot take a larger sample than %d" % len(self.msg_ids)
        return {"msg_ids": resp}


api.add_resource(Messages, "/messages")
logger.info("App initialized")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
