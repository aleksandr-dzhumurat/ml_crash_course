"""
----------------------------
Web-service with API
----------------------------
"""
import os
import unicodedata


import numpy as np
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse

from src.utils import conf, logger, MessagesDB

db = MessagesDB(conf)
db.init_db()

def load_model(model_path):
    # ------ YOUR CODE HERE ----------- #
    # load trained model

    # --------------------------------- #
    return None


app = Flask(__name__)
api = Api(app)
model = load_model(conf.model_path)


@app.route('/messages/<string:identifier>')
def pridict_labell(identifier):
    msg = db.read_message(msg_id=int(identifier))
    # ------ YOUR CODE HERE ----------- #
    # model predict single label

    predicted_label = 'Error: Model not loaded'

    # --------------------------------- #
    return render_template('page.html', id=identifier, txt=msg['txt'], label=predicted_label)

@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # ------ YOUR CODE HERE ----------- #
    # rank all messages and predict

    predicted_label = 'Error: Model not loaded'

    # --------------------------------- #
    recs = [
        {'msg_id': 1, 'msg_txt': 'example_txt_1'},
        {'msg_id': 2, 'msg_txt': 'example_txt_2'}
    ]
    return render_template('feed.html', recs=recs)

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
