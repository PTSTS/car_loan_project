import json
from flask import Flask, request, jsonify
import pandas as pd
from preprocess import preprocess_pipeline
from predict import load_model, predict
from database import connect
import psycopg2


app = Flask(__name__)

@app.route('/', methods=['POST'])
def make_prediction():
    features = json.loads(request.data)
    prediction = predict(model, features)
    return jsonify(prediction)


@app.route('/register', methods=['POST'])
def register():
    data = json.loads(request.data)
    sql = f"""SELECT * FROM users WHERE username = {data['username']}"""

    cursor = connection.cursor()
    response = -1
    try:
        cursor.execute(sql)
        connection.commit()
        if len(cursor.fetchall()):
            response = -1
        else:
            response = 0
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    sql = f"""INSERT INTO users (username, password) VALUES (
      '{data['username']}',
      crypt('{data['password']}', gen_salt('bf'))
    );"""
    try:
        cursor.execute(sql)
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    cursor.close()
    return jsonify(response)


if __name__ == '__main__':
    # model is loaded in __main__
    connection = connect()

    path = 'data/car_loan_trainset.csv'
    model_save_path = 'saved/model_no_id'
    df = pd.read_csv(path).tail(100)
    df, x_columns = preprocess_pipeline(df)

    model, optimizer = load_model(model_save_path, len(x_columns), lr=0.000001)
    app.run(debug=True)
