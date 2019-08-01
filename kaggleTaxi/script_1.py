import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np

# from utils import read_csv, write_csv

BATCH_SIZE = 10000


class TestNet(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(TestNet, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden_1)
        self.l2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.l3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.l4 = nn.Linear(n_hidden_3, n_output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


month_to_season = [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]

df = pd.DataFrame(pd.read_csv('file/set10w.csv', header=0))
df.rename(columns={'fare_amount': 'fare',
                   'pickup_datetime': 'datetime',
                   'pickup_longitude': 'slong',
                   'pickup_latitude': 'slati',
                   'dropoff_longitude': 'elong',
                   'dropoff_latitude': 'elati',
                   'passenger_count': 'pnum',}, inplace = True)

df['year'] = list(int(datetime[:4]) for datetime in df['datetime'])
season = pd.get_dummies(list(month_to_season[int(datetime[5:7]) - 1] for datetime in df['datetime']))
df['spring'], df['summer'], df['autumn'], df['winter'] = season[0], season[1], season[2], season[3]
del season
del df['datetime']
del df['key']

tensor = torch.tensor(df.values, dtype=torch.float)
train_input = tensor[:, 1:]
train_target = tensor[:, :1]
distance = ((train_input[:, 0:1] - train_input[:, 2:3]).pow(2) + (train_input[:, 1:2] - train_input[:, 3:4]).pow(2)).pow(0.5)
train_input = torch.cat((train_input[:, 4:], distance), 1)


torch_dataset = Data.TensorDataset(train_input, train_target)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

tnet = TestNet(7, 15, 20, 10, 1)
optimizer = torch.optim.Adam(tnet.parameters(), lr=0.005)
loss_func = torch.nn.MSELoss()

for epoch in range(100):
    for step, (batch_x, batch_y) in enumerate(loader):
        out = tnet(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        print('epoch: ' + str(epoch) + ' - step: ' + str(step) + ' - RMSE: ' + str(loss.pow(0.5).item()))
