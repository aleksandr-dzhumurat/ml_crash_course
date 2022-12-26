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

db = MessagesDB(conf)
db.init_db()

def load_model(model_path, vec_model_path):
    # ------ YOUR CODE HERE ----------- #
    # load trained model

    model = pickle.load(open(model_path, 'rb'))
    vec_model = pickle.load(open(vec_model_path, 'rb'))
    return model, vec_model

    # --------------------------------- #


app = Flask(__name__)
api = Api(app)
model, vec_model = load_model(conf.model_path, conf.vec_model_path)


@app.route('/messages/<string:identifier>')
def predict_label(identifier):
    msg = db.read_message(msg_id=int(identifier))
    # ------ YOUR CODE HERE ----------- #
    # model predict single label

    msg_csr = vec_model.transform([msg['txt']])
    predicted_label = model.predict(msg_csr)[0]

    # --------------------------------- #
    return render_template('page.html', id=identifier, txt=msg['txt'], label=predicted_label)

@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # ------ YOUR CODE HERE ----------- #
    # rank all messages and predict

    scored_msgs = []
    msg_ids = db.get_messages_ids()
    for i in msg_ids:
        msg = db.read_message(i)
        msg_csr = vec_model.transform([msg['txt']])
        predicted_proba = model.predict_proba(msg_csr)[0][0]
        scored_msgs.append((predicted_proba, msg))
    scored_msgs.sort(key=lambda x: x[0])
    recs = [msg[1] for msg in scored_msgs[:limit]]
    return render_template('feed.html', recs=recs)

    # --------------------------------- #

class Messages(Resource):
    def __init__(self):
        super(Messages, self).__init__()
        self.msg_ids = db.get_messages_ids()  # type: list

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('limit', type=int, default=10, location='args')
        args = parser.parse_args()
        try:
            resp = [int(i) for i in np.random.choice(self.msg_ids, size=args.limit, replace=False)]
        except ValueError as e:
            resp = 'Error: Cannot take a larger sample than %d' % len(self.msg_ids)
        return {'msg_ids': resp}


api.add_resource(Messages, '/messages')
logger.info('App initialized')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
