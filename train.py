import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import pickle
from preprocess import one_hot_encode


class Model(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 128)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(128, 16)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(16, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
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



def train(x_train, y_train, x_test, y_test, optimizer=Adam, epochs=500, lr=0.01):
    train_data_loader = DataLoader(CarLoanDataset(x_train, y_train), batch_size=32, shuffle=True)
    test_data_loader = DataLoader(CarLoanDataset(x_test, y_test), batch_size=32, shuffle=True)
    model = Model(len(x_train.columns))
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_loss_record = []
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            target = batch[1].long()
            output = model.forward(batch[0])
            loss = loss_function(output, target)
            epoch_loss += loss
            # print(f'\tBatch #{i} loss:{int(loss.item())}')

            epoch_loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
            for f in model.parameters():
                f.data.sub_(f.grad.data * lr)
            losses.append(epoch_loss_record)
        print('Epoch loss: %f' %(epoch_loss / len(train_data_loader)))
        test(model, loss_function, x_train, y_train, x_test, y_test)
    return model, optimizer


def test(model, loss_function, x_train, y_train, x_test, y_test):
    test_data_loader = DataLoader(CarLoanDataset(x_test, y_test))

    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for test in test_data_loader:
            x, y = test[0], test[1]
            outputs = model.forward(x)
            loss = loss_function(outputs, y.long())
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print('Val loss: %f' % (total_loss / len(test_data_loader)))
    print('Accuracy: %f %%' % (100 * correct / total))


def predict(model, x):
    x = torch.tensor(x).float()
    outputs = model.forward(x)
    _, prediction = torch.max(outputs.data, 1)
    return int(prediction)


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
    model_save_path = 'saved/model'
    df = pd.read_csv(path)

    for col in df.columns:
        mask = df[col] != np.inf
        df.loc[~mask, col] = df.loc[mask, col].max()

        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[~mask, col].min()

    df = df.drop('customer_id', 1)


    # encoded = df
    # print(encoded)

    # df_pca, pca_model = pca(encoded)
    # print(df_pca)
    # del encoded['customer_id']
    #

    encoded, encoder = one_hot_encode(df, categorical_columns)

    numerical_columns = [x for x in encoded.columns if x not in categorical_columns]
    y_label = 'loan_default'
    x_columns = list(encoded[numerical_columns])
    x_columns.remove(y_label)

    x_train, x_test, y_train, y_test = train_test_split(encoded[x_columns], encoded[y_label])

    model, optimizer = train(x_train, y_train, x_test, y_test, lr=0.000001, epochs=50)
    save(model, optimizer, model_save_path)
