import pandas as pd
import requests, json
from preprocess import preprocess_pipeline

url = 'http://127.0.0.1:5000/'
"""
/predict: POST {'feature1': 'value1', 'feature2': 'value2', ...} (must be logged in to predict) Response prediction
/register: POST {'username': ..., 'password': ***} Response status (success, already exists or error)
/login: POST {'username': ..., 'password': ***} Response status (logged in, wrong or error)
/predict_no_login: POST {'feature1': 'value1', 'feature2': 'value2', ...} (debug, no login needed) Response prediction
"""


def register(username, password):
    response = requests.post(
        url + 'register',
        json.dumps({'username': username, 'password': password})
    )
    print(response.text)


def login(username, password, session):
    response = session.post(
        url + 'login',
        json.dumps({'username': username, 'password': password})
    )
    print(response.text)


def predict(features, session, need_login=True):
    for feature in features:
        data = json.dumps(list(feature))
        if need_login:
            response = session.post(url + 'predict', data)
        else:
            response = session.post(url + 'predict_no_login', data)
        print(response, response.text)


if __name__ == '__main__':
    # response = requests.post(
    #     url + 'register',
    #     json.dumps({'username': 'newuser', 'password': 'newpassword'})
    # )
    # print(response.text)
    #
    # exit()


    # login('abc', 'abc')
    # exit()
    # path = 'data/car_loan_trainset.csv'
    # model_save_path = 'saved/model_no_id'
    # df = pd.read_csv(path).tail(3)
    #
    # df, x_columns = preprocess_pipeline(df)
    # features = df[x_columns].to_numpy()
    session = requests.session()
    login('abc', 'abc', session)
    login('abc', 'abc', session)
    # predict(features, need_login=True)
