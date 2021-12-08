from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train = train.drop(['id', 'contents_open_dt'], axis=1)

test = test.drop(['id', 'contents_open_dt'], axis=1)


model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)
preds = model.predict(test)


submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds

submission.to_csv('baseline.csv', index=False)