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
import pickle

db = MessagesDB(conf)
db.init_db()

def load_model(model_path):
    # ------ YOUR CODE HERE ----------- #
    # load trained model
    loaded_model, loaded_vectorizer = pickle.load(open(model_path,'rb'))
    # --------------------------------- #
    return loaded_model, loaded_vectorizer


app = Flask(__name__)
api = Api(app)


@app.route('/messages/<string:identifier>')
def pridict_labell(identifier):
    msg = db.read_message(msg_id=int(identifier))
    # ------ YOUR CODE HERE ----------- #
    # model predict single label

    model, vectorizer = load_model(conf.model_path)
    vectorized = vectorizer.transform([msg['txt']])
    predicted_label = model.predict(vectorized)[0] 
    predicted_score = model.predict_proba(vectorized)[0][1] 

    # --------------------------------- #
    return render_template('page.html', id=identifier, txt=msg['txt'], label=predicted_label, score=predicted_score)

@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # ------ YOUR CODE HERE ----------- #
    # rank all messages and predict

    model,vectorizer = load_model(conf.model_path)
    
    all_msgs = np.array([{'msg_id': i[0], 'msg_txt': i[1]} for i in db.get_all_messages()])
    vectorized = vectorizer.transform([msg['msg_txt'] for msg in all_msgs])
    predicted_score = [two_scores[0] for two_scores in model.predict_proba(vectorized)] 
    predicted_score = np.array(predicted_score)
    inds = predicted_score.argsort()
    
    sorted_msgs = all_msgs[inds]
    return render_template('feed.html', recs=sorted_msgs[:limit])
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
