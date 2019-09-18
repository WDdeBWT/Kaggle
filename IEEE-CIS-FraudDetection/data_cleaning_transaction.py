import time
import pickle

import numpy as np
import pandas as pd


df_train_tr = None
with open('reduce_mem_train_transaction.pkl', 'rb') as rb:
    df_train_tr = pickle.load(rb)
len_df_train_tr = len(df_train_tr)

df_test_tr = None
with open('reduce_mem_test_transaction.pkl', 'rb') as rb:
    df_test_tr = pickle.load(rb)
len_df_test_tr = len(df_test_tr)

is_fraud = df_train_tr['isFraud']
del df_train_tr['isFraud']
df_tr = pd.concat([df_train_tr, df_test_tr])

for col_name, series in df_tr.iteritems():
    if series.dtype == object:
        if len(series.value_counts()) <= 5:
            dummy_df = pd.get_dummies(series).astype(np.float32)
            for index, dummy_series in dummy_df.iteritems():
                df_tr[col_name + '_' + index] = dummy_series
            del df_tr[col_name]
        else:
            if col_name == 'P_emaildomain':
                df_tr['P_emaildomain'] = df_tr['P_emaildomain'].apply(lambda x: x if x in ['anonymous.com', 'aol.com', 'comcast.net', 'gmail.com', 'hotmail.com', 'icloud.com', 'outlook.com', 'yahoo.com'] else 'None')
                dummy_df = pd.get_dummies(series).astype(np.float32)
                for index, dummy_series in dummy_df.iteritems():
                    df_tr[col_name + '_' + index] = dummy_series
                del df_tr[col_name]
            elif col_name == 'R_emaildomain':
                df_tr['R_emaildomain'] = df_tr['R_emaildomain'].apply(lambda x: x if x in ['anonymous.com', 'aol.com', 'gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com'] else 'None')
                dummy_df = pd.get_dummies(series).astype(np.float32)
                for index, dummy_series in dummy_df.iteritems():
                    df_tr[col_name + '_' + index] = dummy_series
                del df_tr[col_name]
            else:
                raise RuntimeError('col_name unknown: ' + col_name)

for col_name, series in df_tr.iteritems():
    if series.isnull().any():
        df_tr[col_name + '_isnull'] = series.isnull()
        df_tr[col_name] = series.fillna(series.mean())

print('------------------------------')
print(df_tr)
print('------------------------------')
print(df_tr.dtypes)
print('------------------------------')
print(df_tr.info())
print('------------------------------')

df_train_tr = df_tr[:len_df_train_tr]
df_test_tr = df_tr[len_df_train_tr:]
assert len(df_test_tr) == len_df_test_tr

with open('processed_train_transaction.pkl', 'wb') as wb:
    pickle.dump(df_train_tr, wb)
with open('processed_test_transaction.pkl', 'wb') as wb:
    pickle.dump(df_test_tr, wb)
