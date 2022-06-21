import torch
import pandas as pd
import numpy as np
from train import categorical_columns, Model, Adam
from torch.nn.functional import binary_cross_entropy
import pickle
import psycopg2
from database import connect, type_conversion, update_values, insert_row


encoded_data_path = 'data/car_loan_encoded.csv'
model_save_path = 'saved/model_no_id'
one_hot_encoder_path = 'saved/encoder.pkl'


def predict(model, x):
    x = torch.tensor(x).float().to(model.device)

    with torch.no_grad():
        outputs = model.forward(x)
    # _, prediction = torch.max(outputs.data, 1)
    return outputs.item()


def load_model(path, n_features, lr):
    model = Model(n_features)
    optimizer = Adam(model.parameters(), lr=lr)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    return model, optimizer


def create_prediction_table(name, df):
    columns = df.columns
    sql = f"""CREATE TABLE {name} ({columns[0]} {type_conversion[str(df[columns[0]].dtype)]} NOT NULL,
    """
    for column in columns[1:]:
        sql += f"""\t{column} {type_conversion[str(df[column].dtype)]},\n"""
    sql += f"PRIMARY KEY ({columns[0]}) )"
    connection = connect()
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def insert_predictions(all_predictions):
    connection = connect()
    for row in all_predictions.values.tolist():
        keys = all_predictions.columns
        insert_row(connection, 'prediction', keys, row)


if __name__ == '__main__':
    path = 'data/car_loan_trainset.csv'
    # encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
    df = pd.read_csv(path)
    y_label = 'loan_default'
    for col in df.columns:
        mask = df[col] != np.inf
        df.loc[~mask, col] = df.loc[mask, col].max()  # replace inf with max value

        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[~mask, col].min()  # replace NaN with min value

    # encoded_data = encoder.transform(df[categorical_columns]).toarray()
    df = df.drop(categorical_columns, 1)
    # df = pd.concat(
    #     [df, pd.DataFrame(encoded_data)],
    #     axis=1
    # )

    numerical_columns = [x for x in df.columns if x not in categorical_columns and x != 'customer_id']
    x_columns = list(df[numerical_columns])
    x_columns.remove(y_label)

    # df = pd.concat([pd.read_csv(path)['customer_id'], pd.read_csv(encoded_data_path)], 1)
    model, optimizer = load_model(model_save_path, len(x_columns), lr=0.000001)

    print('Predicting...')
    predictions = []
    # for row in df[x_columns].iterrows():
    #     print(row)
    #     predictions.append(predict(model, list(row[1])),)
    all_predictions = pd.concat(
        [
            df['customer_id'],
            pd.DataFrame([predict(model, list(row[1])) for row in df[x_columns].iterrows()]),
            df[y_label],
        ], 1,
    )
    all_predictions.columns = ['customer_id', 'prediction', y_label]
    print(all_predictions)
    print(binary_cross_entropy(torch.tensor(all_predictions['prediction']).double(),
                               torch.tensor(all_predictions[y_label]).double()))

    # connection = connect()
    # for row in all_predictions.values.tolist():
    #     keys = all_predictions.columns
    #
    #     insert_row(connection, 'prediction', keys, row)
    #
    # update_values(
    #     connection,
    #     'prediction',
    #     'customer_id',
    #     all_predictions['customer_id'].astype(int).tolist(),
    #     'prediction',
    #     all_predictions['prediction'].tolist()
    # )
    # update_values(
    #     connection,
    #     'prediction',
    #     'customer_id',
    #     all_predictions['customer_id'].tolist(),
    #     y_label,
    #     all_predictions[y_label].astype(int).tolist()
    # )
    # create_prediction_table('prediction', all_predictions)
