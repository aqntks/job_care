from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import warnings
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', 1000)
warnings.filterwarnings(action='ignore')


train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')

x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

x_train['contents_open_dt'] = x_train.contents_open_dt.apply(pd.to_datetime)
x_train['month'] = x_train.contents_open_dt.apply(lambda x : x.month)
x_test['contents_open_dt'] = x_test.contents_open_dt.apply(pd.to_datetime)
x_test['month'] = x_test.contents_open_dt.apply(lambda x : x.month)

# 컬럼 조합
person_attribute_a_train = x_train['person_attribute_a']
person_attribute_a_1_train = x_train['person_attribute_a_1']
combine_person_attribute_a_train = person_attribute_a_train * 10 + person_attribute_a_1_train
person_attribute_a_test = x_test['person_attribute_a']
person_attribute_a_1_test = x_test['person_attribute_a_1']
combine_person_attribute_a_test = person_attribute_a_test * 10 + person_attribute_a_1_test

x_train['combine_person_attribute_a'] = combine_person_attribute_a_train
x_test['combine_person_attribute_a'] = combine_person_attribute_a_test

## id 제거
x_train = x_train.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
                        'person_attribute_a_1', 'person_attribute_a_1'], axis=1)
x_test = x_test.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
                        'person_attribute_a_1', 'person_attribute_a_1'], axis=1)


x_train = pd.get_dummies(x_train, columns=['person_attribute_b', 'person_prefer_c'])
x_test = pd.get_dummies(x_test, columns=['person_attribute_b', 'person_prefer_c'])


# model = MLPClassifier(hidden_layer_sizes=(30,), learning_rate_init=0.01, max_iter=10, random_state=11)
model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 31, max_depth=-1, learning_rate = 0.1,
                           n_estimators = 950, subsample_for_bin = 100000, objective = None, class_weight = 'balanced',
                           min_split_gain = 0.0, min_child_weight = 0.001, min_child_samples = 150, subsample = 1.0,
                           subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
                           random_state = None, n_jobs = - 1, importance_type = 'split')


model.fit(x_train, y_train)
preds = model.predict(x_test)


submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds

submission.to_csv('csv/lightGBM_v8.csv', index=False)
