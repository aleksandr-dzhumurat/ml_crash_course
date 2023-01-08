"""
----------------------------
Web-service with API
----------------------------
"""
import os
import unicodedata
import pickle

import numpy as np
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse

from src.utils import conf, logger, MessagesDB


def load_model():
    with open(conf.model_path, 'rb') as f:
        return pickle.load(f)


def load_vectorizer():
    with open(conf.vectorizer_path, 'rb') as f:
        return pickle.load(f)


app = Flask(__name__)
api = Api(app)
model = load_model()
vectorizer = load_vectorizer()

db = MessagesDB(conf)
db.init_db(model, vectorizer)


@app.route('/messages/<string:identifier>')
def pridict_labell(identifier):
    msg = db.read_message(msg_id=int(identifier))
    text = msg['txt']
    # ------ YOUR CODE HERE ----------- #
    # model predict single label

    csr = vectorizer.transform([text])
    predicted_label = model.predict(csr)[0]

    # --------------------------------- #
    return render_template('page.html', id=identifier, txt=text, label=predicted_label)


@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # ------ YOUR CODE HERE ----------- #
    # rank all messages and predict

    messages = db.messages(limit)

    # --------------------------------- #
    return render_template('feed.html', recs=messages)


class Messages(Resource):
    def __init__(self):
        super(Messages, self).__init__()
        self.msg_ids = db.get_messages_ids()  # type: list

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('limit', type=int, default=10, location='args')
        args = parser.parse_args()
        try:
            resp = [int(i) for i in np.random.choice(
                self.msg_ids, size=args.limit, replace=False)]
        except ValueError as e:
            resp = 'Error: Cannot take a larger sample than %d' % len(
                self.msg_ids)
        return {'msg_ids': resp}


api.add_resource(Messages, '/messages')
logger.info('App initialized')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
