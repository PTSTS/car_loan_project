import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from preprocess import preprocess_pipeline
from sklearn.utils.class_weight import compute_class_weight


class Model(nn.Module):
    """
    A simple neural network, output for binary classification, can run on GPU
    linear[x, 64]->relu->dropout(50%)->linear[64, 16]->relu->dropout(50%)->linear[16, 1]->sigmoid->out(1)
    """

    def __init__(self, n_features):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(n_features, 64, device=self.device)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(64, 16, device=self.device)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(16, 1, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.to(self.device)
        X = relu(self.linear1(X))
        X = self.dropout1(X)
        X = relu(self.linear2(X))
        X = self.dropout2(X)
        X = self.linear3(X)
        return self.sigmoid(X)


class CarLoanDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y):
        self.X = torch.tensor(X.to_numpy()).float()
        self.y = torch.tensor(y.to_numpy()).float()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)



def train(x_train, y_train, x_test, y_test, optimizer=Adam, epochs=500, lr=0.01, verbose=True):
    train_data_loader = DataLoader(CarLoanDataset(x_train, y_train), batch_size=32, shuffle=True)
    model = Model(len(x_train.columns))
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_function = torch.nn.functional.binary_cross_entropy
    losses = []
    class_weights = compute_class_weight('balanced', [0, 1], y_train)

    for e in range(epochs):
        epoch_loss = 0.0
        epoch_loss_record = []
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            target = batch[1].to(model.device)
            target_weights = torch.tensor([class_weights[int(y.item())] for y in target], dtype=torch.float
                                          ).to(model.device)
            output = model.forward(batch[0]).squeeze()
            loss = loss_function(output, target, weight=target_weights, reduction='mean')
            epoch_loss += loss
            epoch_loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
            for f in model.parameters():
                f.data.sub_(f.grad.data * lr)
            losses.append(epoch_loss_record)
        print('Epoch loss: %f' %(epoch_loss / len(train_data_loader)))
        validate(model, loss_function, x_test, y_test, verbose)
    return model, optimizer


def validate(model, loss_function, x_test, y_test, verbose=True):
    test_data_loader = DataLoader(CarLoanDataset(x_test, y_test))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for test in test_data_loader:
            x, y = test[0], test[1]
            outputs = model.forward(x).squeeze(dim=0)
            loss = loss_function(outputs, y.to(model.device))
            total_loss += loss.item()
            predicted = int(outputs.squeeze().item() > 0.5)
            true = y.item()
            total += y.size(0)
            TP += int((predicted == true) and (predicted == 1))
            TN += int((predicted == true) and (predicted == 0))
            FP += int((predicted != true) and (predicted == 1))
            FN += int((predicted != true) and (predicted == 0))
    if verbose:
        print(' Val loss: %f' % (total_loss / len(test_data_loader)))
        print(f""" F1: {2*TP / (2 * TP + FP + FN)} | ACC: {(TP + TN) / total} | Precision: {TP / (TP + FP)} | Recall:\
{TP / (TP + FN)}""")


def save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


categorical_columns = [
    'branch_id',
    'supplier_id',
    'manufacturer_id',
    'area_id',
    'employee_code_id',
]


if __name__ == '__main__':
    path = 'data/car_loan_trainset.csv'
    encoded_data_path = 'data/car_load_encoded.csv'
    model_save_path = 'saved/model_no_id'
    one_hot_encoder_path = 'saved/encoder.pkl'
    # df = pd.read_csv(path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # for col in df.columns:
    #     mask = df[col] != np.inf
    #     df.loc[~mask, col] = df.loc[mask, col].max()  # replace inf with max value
    #
    #     mask = df[col].isnull()
    #     df.loc[mask, col] = df.loc[~mask, col].min()  # replace NaN with min value
    #
    # df = df.drop('customer_id', 1)
    #
    #
    # encoded = df
    df = pd.read_csv('data/car_loan_trainset.csv')
    df, x_columns = preprocess_pipeline(df, scale=True, scaler_path='saved/scaler.pkl')
    # print(encoded)

    # df_pca, pca_model = pca(encoded)
    # print(df_pca)
    # del encoded['customer_id']
    #

    # encoded, encoder = one_hot_encode(df, categorical_columns)


    # encoded.to_csv(open('data/car_loan_encoded.csv', 'w+', encoding='utf-8'))
    # pickle.dump(encoder, open(one_hot_encoder_path, 'wb+'))

    y_label = 'loan_default'
    x_train, x_test, y_train, y_test = train_test_split(df[x_columns], df[y_label])

    model, optimizer = train(x_train, y_train, x_test, y_test, lr=0.0000001, epochs=100)
    save(model, optimizer, model_save_path)
