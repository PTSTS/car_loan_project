import json
from flask import Flask, request, jsonify, Response, session, redirect
import pandas as pd
from preprocess import preprocess_pipeline
from predict import load_model, predict
from database import connect
import psycopg2
from flask_login import current_user, login_user, UserMixin, LoginManager, login_required


login_manager = LoginManager()
app = Flask(__name__)
app.secret_key = 'abc'
login_manager.init_app(app)

authenticated_users = []


@app.route('/predict', methods=['POST'])
def make_prediction():
    """
    Predict with features, data: {'feature_1': value_1,...}, user authentication needed.
    :return: if logged in: predicted value, else error response
    """
    if current_user.is_authenticated:
        features = json.loads(request.data)
        prediction = predict(model, features)
        submit_prediction(prediction, features)
        return jsonify(prediction)
    return Response('Please login first!', 401)


@app.route('/predict_no_login', methods=['POST'])
def make_prediction_no_login():
    """
    Predict with features, data: {'feature_1': value_1,...}, user authentication not needed.
    :return: predicted value
    """
    features = json.loads(request.data)
    prediction = predict(model, features)
    submit_prediction(prediction, features)
    return jsonify(prediction)


@app.route('/register', methods=['POST', ])
def register():
    """
    Register user in the DB
    :return: response
    """
    data = json.loads(request.data)
    sql = f"""SELECT * FROM users WHERE username = '{data['username']}';"""

    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        connection.commit()
        if len(cursor.fetchall()):
            return Response('Username already exists, choose a different one!', 401)
        else:
            sql = f"""INSERT INTO users (username, password) VALUES (
              '{data['username']}',
              crypt('{data['password']}', gen_salt('bf'))
            );"""
            try:
                cursor.execute(sql)
                connection.commit()
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                return Response('Sorry, server error.', 404)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return Response('Sorry, server error.', 404)
    cursor.close()
    return Response('User registration successful!', 200)


@app.route('/login', methods = ['POST'])
def login():
    """
    Login, data: {'username':..., 'password':,,,}
    :return: response
    """
    data = json.loads(request.data)
    if current_user.is_authenticated:
        return Response('Already logged in.', 200)
    cursor = connection.cursor()
    sql = f"""SELECT * FROM users WHERE username = '{data['username']}'
    AND password = crypt('{data['password']}', password);"""

    try:
        cursor.execute(sql)
        connection.commit()
        if len(cursor.fetchall()):
            user = load_user(data['username'])
            login_user(user, remember=True)
            session['logged_in'] = True
            return Response('Logged in', 200)
        else:
            return Response('Wrong user credentials', 401)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return Response('Sorry, server error.', 404)


def submit_prediction(prediction, features, customer_id=None):
    """
    Submit prediction to the DB
    :param float prediction: prediction to be stored
    :param iterable features: list of features, must be numerical
    :param Optional[int] customer_id:
    """
    cursor = connection.cursor()
    if customer_id is not None:
        sql = f"""INSERT INTO user_prediction (customer_id, prediction, features, time_predicted)
        VALUES ({customer_id}, {prediction}, '{{{', '.join(features)}}}', now())"""
    else:
        sql = f"""INSERT INTO user_prediction (prediction, features, time_predicted)
        VALUES ({prediction}, '{{{', '.join([str(x) for x in features])}}}', now())"""

    cursor.execute(sql)
    connection.commit()
    cursor.close()


@login_manager.user_loader
def load_user(id):
    for user in  authenticated_users:
        if user.id == id:
            return user
    return User(id, None)


class User():
    def __init__(self, username, hash):
        self.username = username

    @property
    def id(self):
        return self.username

    @property
    def is_authenticated(self):
        return True
    #
    def is_active(self):
        return self.is_active

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return self.username


if __name__ == '__main__':
    connection = connect()

    path = 'data/car_loan_trainset.csv'
    model_save_path = 'saved/model_no_id'
    df = pd.read_csv(path).head(1)
    df, x_columns = preprocess_pipeline(df)

    model, optimizer = load_model(model_save_path, len(x_columns), lr=0.000001)

    app.run(debug=False)

