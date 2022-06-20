import pandas as pd
import requests, json
from preprocess import preprocess_pipeline

url = 'http://127.0.0.1:5000/'

if __name__ == '__main__':
    path = 'data/car_loan_trainset.csv'
    model_save_path = 'saved/model_no_id'
    df = pd.read_csv(path).tail(100)

    df, x_columns = preprocess_pipeline(df)
    features = df[x_columns].to_numpy()
    for row in features:
        data = json.dumps(list(row))
        response = requests.post(url, data)
        print(response, response.text)
