import pandas as pd
import numpy as np
import math
import torch

month_to_season = [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]

df = pd.DataFrame(pd.read_csv('file/set1w.csv', header=0))
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
# df['distance'] = list(m for slong, slati, elong, elati in df['slong'], df['slati'], df['elong'], df['elati'])

tensor = torch.tensor(df.values)
print(tensor)
train_input = tensor[:, 1:]
train_target = tensor[:, :1]
print(train_input)
print(train_target)



