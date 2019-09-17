import pickle
import numpy as np
import pandas as pd


def fileter_numeric_cols(data):

    numeric_cols = set()
    numric_cols_cnt = 0

    for col in data.columns:
        col_type = str(data[col].dtypes)
        #print(col_type, type(col_type))
        if col_type[:3] == 'int' or col_type[:5] == 'float' or col_type[:6] == 'double':
            numeric_cols.add(col_type)
            numric_cols_cnt  += 1

    return list(numeric_cols), numric_cols_cnt, numric_cols_cnt/len(data.columns)


def reduce_mem_usage(df, numerics, verbose=True):
    #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            assert c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max, 'Value Error'
            df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, \
                                                                    100 * (start_mem - end_mem) / start_mem))
    return df


if __name__ == "__main__":
    pd_reader1 = pd.read_csv('file/train_transaction.csv') # TODO: change file path
    df_train = pd.DataFrame(pd_reader1)
    pd_reader2 = pd.read_csv('file/test_transaction.csv') # TODO: change file path
    df_test = pd.DataFrame(pd_reader2)
    train_num_cols, train_numric_cols_cnt, train_numric_cols_rate = fileter_numeric_cols(df_train)
    df_train = reduce_mem_usage(df_train, train_num_cols)
    with open('reduce_mem_train_transaction.pkl', 'wb') as wb: # TODO: change file path
        pickle.dump(df_train, wb)
    df_test = reduce_mem_usage(df_test, train_num_cols)
    with open('reduce_mem_test_transaction.pkl', 'wb') as wb: # TODO: change file path
        pickle.dump(df_test, wb)

    pd_reader1 = pd.read_csv('file/train_identity.csv') # TODO: change file path
    df_train = pd.DataFrame(pd_reader1)
    pd_reader2 = pd.read_csv('file/test_identity.csv') # TODO: change file path
    df_test = pd.DataFrame(pd_reader2)
    train_num_cols, train_numric_cols_cnt, train_numric_cols_rate = fileter_numeric_cols(df_train)
    df_train = reduce_mem_usage(df_train, train_num_cols)
    with open('reduce_mem_train_identity.pkl', 'wb') as wb: # TODO: change file path
        pickle.dump(df_train, wb)
    df_test = reduce_mem_usage(df_test, train_num_cols)
    with open('reduce_mem_test_identity.pkl', 'wb') as wb: # TODO: change file path
        pickle.dump(df_test, wb)
