from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np


from sklearn.metrics import f1_score

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')


# 데이터 로드

train = pd.read_csv('train.csv')      # 총 501951 개   35개 피쳐
test = pd.read_csv('test.csv')

x = train.iloc[:, :-1]
y = train.iloc[:, -1]


# 데이터 확인
# print(train.head())
# print(test.head())
# print(train.shape)
# print(train.describe())
# print(train.info())

# train_count =train['Pclass'].value_counts()
# titanic_df.sort_values(by='Pclass', ascending=True)


# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# 필요 없는 인덱스 칼럼 제거

## id 제거
x_train = x_train.drop(['id', 'contents_open_dt'], axis=1)
x_test = x_test.drop(['id', 'contents_open_dt'], axis=1)
train = train.drop(['id', 'contents_open_dt'], axis=1)
test = test.drop(['id', 'contents_open_dt'], axis=1)

# 결측치 제거
## null 피쳐 없음


# 이상치 제거


# 인코딩
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# pd.set_option('display.max_columns', 1000)
# print(pd.get_dummies(train['person_attribute_a_1']))


# 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaled_data = StandardScaler().fit_transform(input_data)
# scaled_data = MinMaxScaler().fit_transform(input_data)
# amount_n = np.log1p(df_copy['Amount'])
# var_2 = np.expm1(var_1)


# 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier

# model = DecisionTreeClassifier(max_depth=60)
# F1Score -> 0.5540630561659227

# model = LogisticRegression()
# F1Score -> 0.4513787932554173

# model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)
# F1Score -> 0.6339364303178483

rf_model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=300, random_state=11)
model = VotingClassifier(estimators=[('RF', rf_model), ('MLP', mlp_model)], voting='soft')
# F1Score -> 0.6415912229553477

# model = KNeighborsClassifier(n_neighbors=8)
# F1Score -> 0.4790223398458784 -> 오래걸림

# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
# F1Score -> 0.6176210607225211

# model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=300, random_state=11)
# F1Score -> 0.6675647695901467
# model = MLPClassifier(hidden_layer_sizes=(60,), learning_rate_init=0.01, max_iter=300, random_state=11)
# F1Score -> 0.6675647695901467
# model = MLPClassifier(hidden_layer_sizes=(60,), learning_rate_init=0.001, max_iter=300, random_state=11)
# F1Score -> 0.00011928192282459593
# model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate_init=0.01, max_iter=10, random_state=11)

# model = SVC(C=100, gamma=1, random_state=11, probability=True)
# -> 오래걸림
# model = XGBClassifier(n_estimators=300, random_state=11)

# model.fit(x_train, y_train)
# preds = model.predict(x_test)


# 파라미터 튜닝
# GridSearchCV

# 평가   -> F1 Score
# from sklearn.metrics import f1_score
# f1 = f1_score(y_test, preds)
#
# print(f1)


# 제출용
model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=300, random_state=11)
model.fit(x, y)
preds = model.predict(test)

# # 저장
# submission = pd.read_csv('sample_submission.csv')
# submission['target'] = preds
#
# submission.to_csv('now.csv', index=False)
#



