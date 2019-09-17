import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np

import utils

BATCH_SIZE = 2000000
CROSS_VALIDATION_NUM = 5


class TestNet(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(TestNet, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden_1)
        self.l2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.l3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.l4 = nn.Linear(n_hidden_3, n_output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


month_to_season = [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]

def csv_reader(read_path='file/train.csv', header=0, chunksize=3000000):
    pd_reader = pd.read_csv(read_path, header=header, chunksize=chunksize)
    for chunk_index, chunk in enumerate(pd_reader):
        if chunk_index >= 2:
            return
        df = pd.DataFrame(chunk)
        df.rename(columns={'fare_amount': 'fare',
                        'pickup_datetime': 'datetime',
                        'pickup_longitude': 'slong',
                        'pickup_latitude': 'slati',
                        'dropoff_longitude': 'elong',
                        'dropoff_latitude': 'elati',
                        'passenger_count': 'pnum',}, inplace = True)

        # df['year'] = list(int(datetime[:4]) for datetime in df['datetime'])
        season = pd.get_dummies(list(month_to_season[int(datetime[5:7]) - 1] for datetime in df['datetime']))
        season.index = range(chunk_index * chunksize, chunk_index * chunksize + len(season))
        df['spring'], df['summer'], df['autumn'], df['winter'] = season[0], season[1], season[2], season[3]
        del season
        del df['datetime']
        del df['key']

        df['abs_diff_longitude'] = (df.slong - df.elong).abs()
        df['abs_diff_latitude'] = (df.slati - df.elati).abs()
        df = df[(df.abs_diff_longitude<5) & (df.abs_diff_latitude<5)]
        del df['abs_diff_longitude']
        del df['abs_diff_latitude']
        print('csv_reader: ', chunk_index)
        yield torch.tensor(df.values, dtype=torch.float)

train_data = torch.cat([partial_tensor for partial_tensor in csv_reader()])

train_input = train_data[:, 1:]
train_target = train_data[:, :1]
distance = ((train_input[:, 0:1] - train_input[:, 2:3]).abs() + (train_input[:, 1:2] - train_input[:, 3:4]).abs()) * 1000
train_input[:, 3:4] = distance
train_input = train_input[:, 3:]
train_torch_dataset = Data.TensorDataset(train_input, train_target)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

tnet = TestNet(6, 15, 20, 10, 1).cuda()
optimizer = torch.optim.Adam(tnet.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
loss_func = torch.nn.MSELoss()

for epoch in range(1):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = tnet(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        print('epoch: ' + str(epoch) + ' - step: ' + str(step) + ' - RMSE: ' + str(loss.pow(0.5).item()))
    scheduler.step()

# gen result
test_path = 'file/test.csv'
test_df = pd.DataFrame(pd.read_csv('file/test.csv', header=0))
test_df.rename(columns={'pickup_datetime': 'datetime',
                'pickup_longitude': 'slong',
                'pickup_latitude': 'slati',
                'dropoff_longitude': 'elong',
                'dropoff_latitude': 'elati',
                'passenger_count': 'pnum',}, inplace = True)

# test_df['year'] = list(int(datetime[:4]) for datetime in test_df['datetime'])
season = pd.get_dummies(list(month_to_season[int(datetime[5:7]) - 1] for datetime in test_df['datetime']))
test_df['spring'], test_df['summer'], test_df['autumn'], test_df['winter'] = season[0], season[1], season[2], season[3]
del season
del test_df['datetime']
test_keys = test_df['key'].values
del test_df['key']
test_input = torch.tensor(test_df.values, dtype=torch.float)
distance = ((test_input[:, 0:1] - test_input[:, 2:3]).abs() + (test_input[:, 1:2] - test_input[:, 3:4]).abs()) * 1000
test_input[:, 3:4] = distance
test_input = test_input[:, 3:]

test_input = test_input.cuda()
test_out = tnet(test_input)
write_list = [['key', 'fare_amount'],]
print('test_keys', test_keys.shape)
print('test_out', test_out.shape)
print('test_out[0]', test_out[0])
for key, out in zip(test_keys, test_out):
    write_list.append([key, out.item()])
print(len(write_list))
print(write_list[0])
utils.write_csv('result.csv', write_list)