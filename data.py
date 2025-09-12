import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import lib

train_features = pd.read_csv("data/train_features.csv", index_col="id")
test_features = pd.read_csv("data/test_features.csv", index_col="id")
train_labels = pd.read_csv("data/train_labels.csv", index_col="id")
train_labels_refined = pd.read_csv("data/train_labels_refined.csv", index_col="id")

assert all(train_labels.index == train_labels_refined.index)
assert all(train_labels.columns == train_labels_refined.columns)

assert train_labels[train_labels.sum(axis=1) != 1].shape[0] == 0
assert train_labels_refined[train_labels_refined.sum(axis=1) != 1].shape[0] == 0

species_labels = sorted(train_labels.columns.unique())

train_features['resolution'] = train_features['filepath'].apply(lambda filename: lib.get_resolution('data/' + filename))

train_features['site_plus_resolution'] = train_features['site'] + '_' + train_features['resolution']

y = train_labels
y_refined = train_labels_refined
x = train_features.loc[y.index]

val_sites = ['S0060','S0063','S0043','S0038','S0120','S0014']

mask_val  = x['site'].isin(val_sites)

x_train = x[ ~mask_val ]
y_train = y[ ~mask_val ]
y_train_refined = y_refined[ ~mask_val ]

x_eval = x[ mask_val ]
y_eval = y[ mask_val ]
y_eval_refined = y_refined[ mask_val ]

train_labels['label'] = train_labels.to_numpy().argmax(axis=1)

train_all = train_features.merge(train_labels[['label']], on='id')

train_all['fold'] = -1

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(splitter.split(train_all, train_all['label'], groups=train_all['site'])):
    train_all.iloc[val_idx, train_all.columns.get_loc('fold')] = fold
