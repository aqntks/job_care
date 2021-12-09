from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import warnings
from lightgbm import LGBMClassifier


warnings.filterwarnings(action='ignore')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train['contents_open_dt'] = train.contents_open_dt.apply(pd.to_datetime)
train['month'] = train.contents_open_dt.apply(lambda x : x.month)
test['contents_open_dt'] = test.contents_open_dt.apply(pd.to_datetime)
test['month'] = test.contents_open_dt.apply(lambda x : x.month)

train = train.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g'], axis=1)
test = test.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g'], axis=1)

train = pd.get_dummies(train, columns=['person_attribute_b', 'person_prefer_c'])
test = pd.get_dummies(test, columns=['person_attribute_b', 'person_prefer_c'])


# model = MLPClassifier(hidden_layer_sizes=(30,), learning_rate_init=0.01, max_iter=10, random_state=11)
model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 31, max_depth=-1, learning_rate = 0.1,
                           n_estimators = 950, subsample_for_bin = 100000, objective = None, class_weight = 'balanced',
                           min_split_gain = 0.0, min_child_weight = 0.001, min_child_samples = 150, subsample = 1.0,
                           subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
                           random_state = None, n_jobs = - 1, importance_type = 'split')

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)
preds = model.predict(test)


submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds

submission.to_csv('lightGBM_v2.csv', index=False)
