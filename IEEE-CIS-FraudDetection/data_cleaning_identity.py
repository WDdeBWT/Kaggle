import pickle
import numpy as np
import pandas as pd

df_train_id = None
with open('reduce_mem_train_identity.pkl', 'rb') as rb:
    df_train_id = pickle.load(rb)

df_train_id['id_02'] = df_train_id['id_02'].fillna(value=0)
df_train_id['id_03'] = df_train_id['id_03'].fillna(value=0)
df_train_id['id_04'] = df_train_id['id_04'].fillna(value=0)
df_train_id['id_05'] = df_train_id['id_05'].fillna(value=0)
df_train_id['id_06'] = df_train_id['id_06'].fillna(value=0)
df_train_id['id_07'] = df_train_id['id_07'].fillna(value=0)
df_train_id['id_08'] = df_train_id['id_08'].fillna(df_train_id['id_08'].mean())
df_train_id['id_09'] = df_train_id['id_09'].fillna(value=0)
df_train_id['id_10'] = df_train_id['id_10'].fillna(value=0)

df_train_id['id_11'] = df_train_id['id_11']
df_train_id['id_11'] = df_train_id['id_11'].fillna(df_train_id['id_11'].mean())

df_train_id['id_12'] = df_train_id['id_12'].map({'Found': 1, 'NotFound': 0})
df_train_id['id_12'] = df_train_id['id_12'].fillna(value=0).astype(np.float32)

df_train_id['id_13'] = df_train_id['id_13']
df_train_id['id_13'] = df_train_id['id_13'].fillna(df_train_id['id_13'].mean())

df_train_id['id_14'] = df_train_id['id_14']
df_train_id['id_14'] = df_train_id['id_14'].fillna(df_train_id['id_14'].mean())

df_train_id['id_15'] = df_train_id['id_15'].map({'Found': 1, 'Unknown': 0, 'New': -1})
df_train_id['id_15'] = df_train_id['id_15'].fillna(value=0).astype(np.float32)

df_train_id['id_16'] = df_train_id['id_16'].map({'Found': 1, 'NotFound': 0})
df_train_id['id_16'] = df_train_id['id_16'].fillna(value=0).astype(np.float32)

df_train_id['id_17'] = df_train_id['id_17']
df_train_id['id_17'] = df_train_id['id_17'].fillna(df_train_id['id_17'].mean())

df_train_id['id_18'] = df_train_id['id_18'].fillna(df_train_id['id_18'].mean())

df_train_id['id_19'] = df_train_id['id_19']
df_train_id['id_19'] = df_train_id['id_19'].fillna(df_train_id['id_19'].mean())

df_train_id['id_20'] = df_train_id['id_20']
df_train_id['id_20'] = df_train_id['id_20'].fillna(df_train_id['id_20'].mean())

df_train_id['id_21'] = df_train_id['id_21'].fillna(df_train_id['id_21'].mean())
df_train_id['id_22'] = df_train_id['id_22'].fillna(df_train_id['id_22'].mean())

df_train_id['id_23'] = df_train_id['id_23'].map({'IP_PROXY:TRANSPARENT': 1, 'IP_PROXY:ANONYMOUS': 0, 'IP_PROXY:HIDDEN': -1})
df_train_id['id_23'] = df_train_id['id_23'].fillna(df_train_id['id_23'].mean()).astype(np.float32)

df_train_id['id_24'] = df_train_id['id_24'].fillna(df_train_id['id_24'].mean())
df_train_id['id_25'] = df_train_id['id_25'].fillna(df_train_id['id_25'].mean())
df_train_id['id_26'] = df_train_id['id_26'].fillna(df_train_id['id_26'].mean())

df_train_id['id_27'] = df_train_id['id_27'].map({'Found': 1, 'NotFound': 0})
df_train_id['id_27'] = df_train_id['id_27'].fillna(df_train_id['id_27'].mean()).astype(np.float32)

df_train_id['id_28'] = df_train_id['id_28'].map({'Found': 1, 'new': 0})
df_train_id['id_28'] = df_train_id['id_28'].fillna(value=0).astype(np.float32)

df_train_id['id_29'] = df_train_id['id_29'].map({'Found': 1, 'NotFound': 0})
df_train_id['id_29'] = df_train_id['id_29'].fillna(value=0).astype(np.float32)

del df_train_id['id_30']
del df_train_id['id_31']

df_train_id['id_32'] = df_train_id['id_32']
df_train_id['id_32'] = df_train_id['id_32'].fillna(df_train_id['id_32'].mean())

del df_train_id['id_33']

df_train_id['id_34'] = df_train_id['id_34'].map({'match_status:2': 2, 'match_status:1': 1, 'match_status:0': 0, 'match_status:-1': -1})
df_train_id['id_34'] = df_train_id['id_34'].fillna(value=2).astype(np.float32)

df_train_id['id_35'] = df_train_id['id_35'].map({'T': 1, 'F': 0})
df_train_id['id_35'] = df_train_id['id_35'].fillna(df_train_id['id_35'].mean()).astype(np.float32)

df_train_id['id_36'] = df_train_id['id_36'].map({'T': 1, 'F': 0})
df_train_id['id_36'] = df_train_id['id_36'].fillna(value=0).astype(np.float32)

df_train_id['id_37'] = df_train_id['id_37'].map({'T': 1, 'F': 0})
df_train_id['id_37'] = df_train_id['id_37'].fillna(df_train_id['id_37'].mean()).astype(np.float32)

df_train_id['id_38'] = df_train_id['id_38'].map({'T': 1, 'F': 0})
df_train_id['id_38'] = df_train_id['id_38'].fillna(df_train_id['id_38'].mean()).astype(np.float32)

df_train_id['DeviceType'] = df_train_id['DeviceType'].map({'desktop': 1, 'mobile': 0})
df_train_id['DeviceType'] = df_train_id['DeviceType'].fillna(df_train_id['DeviceType'].mean()).astype(np.float32)

del df_train_id['DeviceInfo']

# print('------------------------------')
# print(df_train_id)
# print('------------------------------')
# print(df_train_id.dtypes)
# print('------------------------------')
# print(df_train_id.info())
# print('------------------------------')

df_train_id = df_train_id
np_data = df_train_id.values


print('------------------------------------------------------------')
print(np_data.shape)
print(np_data[:, 1:].shape)

