import pickle

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd

import utils

BATCH_SIZE = 50000


class TestNet(nn.Module):

    def __init__(self, n_input):
        super(TestNet,self).__init__()
        self.fc1 = nn.Linear(n_input, 512)
        self.relu1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.dout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 64)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(64, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout1 = self.dout1(h1)
        a2 = self.fc2(dout1)
        h2 = self.relu2(a2)
        dout2 = self.dout2(h2)
        a3 = self.fc3(dout2)
        h3 = self.relu3(a3)
        dout3 = self.dout3(h3)
        a4 = self.fc4(dout3)
        h4 = self.prelu(a4)
        a5 = self.out(h4)
        y = self.out_act(a5)
        return y

    def predict(self, x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        return torch.tensor(pred)


def prepare_data():

    def judge_zone(dt_hour):
        zone = [(7, 12), (12, 17), (17, 23)]# other is (23, 7)
        for idx, zn in enumerate(zone):
            if dt_hour >= zn[0] and dt_hour < zn[1]:
                return idx
        return 3

    df_train_tr = None
    with open('processed_train_transaction.pkl', 'rb') as rb:
        df_train_tr = pickle.load(rb)

    df_test_tr = None
    with open('processed_test_transaction.pkl', 'rb') as rb:
        df_test_tr = pickle.load(rb)

    df_train_id = None
    with open('processed_train_identity.pkl', 'rb') as rb:
        df_train_id = pickle.load(rb)

    df_test_id = None
    with open('processed_test_identity.pkl', 'rb') as rb:
        df_test_id = pickle.load(rb)

    feature_col = ['TransactionID', 'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7']
    df_train_feature = pd.DataFrame(df_train_id, columns=feature_col)
    df_test_feature = pd.DataFrame(df_test_id, columns=feature_col)

    df_train = pd.merge(df_train_tr, df_train_feature, on=['TransactionID'], how='left')
    df_test = pd.merge(df_test_tr, df_test_feature, on=['TransactionID'], how='left')
    is_fraud = df_train['isFraud'].copy()
    del df_train['isFraud']

    if df_train['feature_0'].isnull().any():
        df_train['feature_isnull'] = df_train['feature_0'].isnull()
        df_train['feature_0'] = df_train['feature_0'].fillna(value=0)
        df_train['feature_1'] = df_train['feature_1'].fillna(value=0)
        df_train['feature_2'] = df_train['feature_2'].fillna(value=0)
        df_train['feature_3'] = df_train['feature_3'].fillna(value=0)
        df_train['feature_4'] = df_train['feature_4'].fillna(value=0)
        df_train['feature_5'] = df_train['feature_5'].fillna(value=0)
        df_train['feature_6'] = df_train['feature_6'].fillna(value=0)
        df_train['feature_7'] = df_train['feature_7'].fillna(value=0)

    if df_test['feature_0'].isnull().any():
        df_test['feature_isnull'] = df_test['feature_0'].isnull()
        df_test['feature_0'] = df_test['feature_0'].fillna(value=0)
        df_test['feature_1'] = df_test['feature_1'].fillna(value=0)
        df_test['feature_2'] = df_test['feature_2'].fillna(value=0)
        df_test['feature_3'] = df_test['feature_3'].fillna(value=0)
        df_test['feature_4'] = df_test['feature_4'].fillna(value=0)
        df_test['feature_5'] = df_test['feature_5'].fillna(value=0)
        df_test['feature_6'] = df_test['feature_6'].fillna(value=0)
        df_test['feature_7'] = df_test['feature_7'].fillna(value=0)

    df_train['TransactionDT'] = df_train['TransactionDT'].apply(lambda x:(x+59) // 60 // 60 % 24)
    df_train['TransactionDT'] = df_train['TransactionDT'].apply(lambda x:judge_zone(x))
    dummy = pd.get_dummies(df_train['TransactionDT'], dummy_na=False, prefix='%s_' % 'TransactionDT')
    df_train = pd.concat((df_train, dummy), axis=1)
    del df_train['TransactionDT']

    df_test['TransactionDT'] = df_test['TransactionDT'].apply(lambda x:(x+59) // 60 // 60 % 24)
    df_test['TransactionDT'] = df_test['TransactionDT'].apply(lambda x:judge_zone(x))
    dummy = pd.get_dummies(df_test['TransactionDT'], dummy_na=False, prefix='%s_' % 'TransactionDT')
    df_test = pd.concat((df_test, dummy), axis=1)
    del df_test['TransactionDT']
    print(df_train)
    return df_train.astype(np.float32).values, df_test.astype(np.float32).values, is_fraud.astype(np.float32).values


def train():
    np_train, np_test, np_is_fraud = prepare_data()
    in_features = np_train.shape[1] - 1

    train_torch_dataset = Data.TensorDataset(torch.tensor(np_train[:, 1:]), torch.tensor(np_is_fraud))
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
    )

    tnet = TestNet(in_features)
    optimizer = torch.optim.Adam(tnet.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    loss_func = torch.nn.BCELoss()

    for epoch in range(10):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x
            batch_y = batch_y
            out = tnet(batch_x)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            print('epoch: ' + str(epoch) + ' - step: ' + str(step) + ' - RMSE: ' + str(loss.pow(0.5).item()))
            scheduler.step()

    return tnet, np_test



tnet, np_test = train()

write_list = [['TransactionID', 'isFraud']]
test_out = tnet(torch.tensor(np_test[:, 1:]))
for id, result in zip(np_test[:, 0].astype(np.int), test_out):
    write_list.append([str(id), float(result)])

utils.write_csv('result0.csv', write_list)

write_list = [['TransactionID', 'isFraud']]
test_out = tnet.predict(tnet(torch.tensor(np_test[:, 1:])))
for id, result in zip(np_test[:, 0].astype(np.int), test_out):
    write_list.append([str(id), float(result)])

utils.write_csv('result1.csv', write_list)
