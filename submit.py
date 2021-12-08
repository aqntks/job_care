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


train = train.drop(['id', 'contents_open_dt'], axis=1)

test = test.drop(['id', 'contents_open_dt'], axis=1)


# model = MLPClassifier(hidden_layer_sizes=(30,), learning_rate_init=0.01, max_iter=10, random_state=11)
model = LGBMClassifier(n_estimators=1000, max_depth= 128, min_child_samples= 100, num_leaves= 64, subsample= 0.8)

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)
preds = model.predict(test)


submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds

submission.to_csv('lightGBM.csv', index=False)
