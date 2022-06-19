# from sklearn.utils import
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from torch.nn import Linear, Dropout


categorical_columns = [
    'branch_id',
    'supplier_id',
    'manufacturer_id',
    'area_id',
    'employee_code_id',
]


def one_hot_encode(df, columns):
    encoder = preprocessing.OneHotEncoder()
    encoded_data = encoder.fit_transform(df[columns]).toarray()
    df =  df.drop(columns, 1)
    df = pd.concat(
        [df, pd.DataFrame(encoded_data)],
        axis=1
    )
    # for column in columns:
    #     integerized_data = preprocessing.LabelEncoder().fit_transform(df[column])
    #     integerized_data = preprocessing.OneHotEncoder().fit_transform(integerized_data.reshape(-1,1)).toarray()
    #
    #     df = pd.concat(
    #         [df, pd.DataFrame(integerized_data)],
    #         axis=1
    #     )
    return df, encoder


def pca(df, n_components=200):
    pca_model = PCA(n_components)
    df = pca_model.fit_transform(df)
    return df, pca_model


if __name__ == '__main__':
    path = 'data/car_loan_trainset.csv'
    df = pd.read_csv(path)

    for col in df.columns:
        mask = df[col] != np.inf
        df.loc[~mask, col] = df.loc[mask, col].max()

        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[~mask, col].min()


    # print(pd.concat([
    #     df.max(axis=0).to_frame(),
    #     df.min(axis=0).to_frame(),
    #     df.mean(axis=0).to_frame(),
    #     df.nunique(axis=0).to_frame()
    # ], axis=1))
    df = df.drop('customer_id', 1)
    # print(df)
    # print((df[df == np.inf]).count())
    # exit()
    # encoded, encoder = one_hot_encode(df, categorical_columns)
    numerical_columns = [x for x in df.columns if x not in categorical_columns]

    # encoded = df
    # print(encoded)

    # df_pca, pca_model = pca(encoded)
    # print(df_pca)
    # del encoded['customer_id']
    #
    y_label = 'loan_default'
    x_columns = list(df[numerical_columns])
    x_columns.remove(y_label)

    x_train, x_test, y_train, y_test = train_test_split(df[x_columns], df[y_label])

    linear_regression_model = LinearRegression().fit(x_train, y_train)
    print(linear_regression_model.predict(x_test))
    print(y_test)
    # print(x_train)
    # print(encoded.shape)
    # print()

