import json
from flask import Flask, request, jsonify
import pandas as pd
from preprocess import preprocess_pipeline
from predict import load_model, predict

app = Flask(__name__)

@app.route('/', methods=['POST'])
def make_prediction():
    features = json.loads(request.data)
    prediction = predict(model, features)
    return jsonify(prediction)


if __name__ == '__main__':
    # model is loaded in __main__

    path = 'data/car_loan_trainset.csv'
    model_save_path = 'saved/model_no_id'
    df = pd.read_csv(path).tail(100)
    df, x_columns = preprocess_pipeline(df)

    model, optimizer = load_model(model_save_path, len(x_columns), lr=0.000001)
    app.run(debug=True)
