import pickle
import numpy as np
import pandas as pd


def get_email_P_info(df_train_tr):
    email_dict_P = {}
    for index, (isFraud, email_P) in enumerate(zip(df_train_tr['isFraud'], df_train_tr['P_emaildomain'])):
        if index % 100000 == 0:
            print('----- ', index)
        if email_P not in email_dict_P:
            email_dict_P[email_P] = [0, 0]
        email_dict_P[email_P][1] += 1
        if isFraud == 1:
            email_dict_P[email_P][0] += 1

    for key in email_dict_P:
        if email_dict_P[key][1] > 5000:
            print('{}: {} / {} ({}%)'.format(key, email_dict_P[key][0], email_dict_P[key][1], (email_dict_P[key][0] * 100) / email_dict_P[key][1]))


def get_email_R_info(df_train_tr):
    email_dict_R = {}
    for index, (isFraud, email_R) in enumerate(zip(df_train_tr['isFraud'], df_train_tr['R_emaildomain'])):
        if index % 100000 == 0:
            print('----- ', index)
        if email_R not in email_dict_R:
            email_dict_R[email_R] = [0, 0]
        email_dict_R[email_R][1] += 1
        if isFraud == 1:
            email_dict_R[email_R][0] += 1

    for key in email_dict_R:
        if email_dict_R[key][1] > 5000:
            print('{}: {} / {} ({}%)'.format(key, email_dict_R[key][0], email_dict_R[key][1], (email_dict_R[key][0] * 100) / email_dict_R[key][1]))


df_train_tr = None
with open('reduce_mem_train_transaction.pkl', 'rb') as rb:
    df_train_tr = pickle.load(rb)

# get_email_P_info(df_train_tr)
get_email_R_info(df_train_tr)
