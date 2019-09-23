import pickle
import pandas as pd


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

print('-----------------------------------')
print(df_train_tr)
print('-----------------------------------')
print(df_test_tr)
print('-----------------------------------')
print(df_train_id)
print('-----------------------------------')
print(df_test_id)
