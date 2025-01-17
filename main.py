from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np


from sklearn.metrics import f1_score

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')


def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True)


mode = True
pd.set_option('display.max_columns', 1000)

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




# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# 필요 없는 인덱스 칼럼 제거

# 컬럼 조합
# person_attribute_a_train = x_train['person_attribute_a']
# person_attribute_a_1_train = x_train['person_attribute_a_1']
# combine_person_attribute_a_train = person_attribute_a_train * 10 + person_attribute_a_1_train
# person_attribute_a_test = x_test['person_attribute_a']
# person_attribute_a_1_test = x_test['person_attribute_a_1']
# combine_person_attribute_a_test = person_attribute_a_test * 10 + person_attribute_a_1_test
#
# x_train['combine_person_attribute_a'] = combine_person_attribute_a_train
# x_test['combine_person_attribute_a'] = combine_person_attribute_a_test
#
#
# ## id 제거
# x_train = x_train.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
#                         'person_attribute_a_1', 'person_attribute_a_1'], axis=1)
# x_test = x_test.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
#                         'person_attribute_a_1', 'person_attribute_a_1'], axis=1)
# train = train.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
#                         'person_attribute_a_1', 'person_attribute_a_1'], axis=1)
# test = test.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g',
#                         'person_attribute_a_1', 'person_attribute_a_1'], axis=1)

# 날짜 처리
# x_train['contents_open_dt'] = x_train.contents_open_dt.apply(pd.to_datetime)
# x_train['month'] = x_train.contents_open_dt.apply(lambda x : x.month)
# x_test['contents_open_dt'] = x_test.contents_open_dt.apply(pd.to_datetime)
# x_test['month'] = x_test.contents_open_dt.apply(lambda x : x.month)

x_train = x_train.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g', 'person_rn', 'contents_rn'], axis=1)
x_test = x_test.drop(['id', 'contents_open_dt', 'person_prefer_f', 'person_prefer_g', 'person_rn', 'contents_rn'], axis=1)

# 결측치 제거
## null 피쳐 없음


# 이상치 제거


# 인코딩
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# 6507
columns = ['person_attribute_b', 'person_prefer_c']
# columns = ['person_attribute_b', 'person_prefer_c', 'person_prefer_e', 'contents_attribute_i',
#            'contents_attribute_a', 'contents_attribute_j_1', 'contents_attribute_j', 'contents_attribute_c',
#            'contents_attribute_k', 'contents_attribute_m', 'contents_attribute_e']
x_train = pd.get_dummies(x_train, columns=columns)
x_test = pd.get_dummies(x_test, columns=columns)

# 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaled_data = StandardScaler().fit_transform(input_data)
# x_train = MinMaxScaler().fit_transform(x_train)
# print(x_train)
# print(np.log1p(x_train['person_prefer_h_1']))
# amount_n = np.log1p(x_train['Amount'])
# var_2 = np.expm1(var_1)


if mode:
    # 모델
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # model = DecisionTreeClassifier(max_depth=60)
    # F1Score -> 0.5540630561659227

    # model = LogisticRegression()
    # F1Score -> 0.4513787932554173

    # model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)
    # F1Score -> 0.6339364303178483

    # rf_model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)
    # mlp_model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=300, random_state=11)
    # model = VotingClassifier(estimators=[('RF', rf_model), ('MLP', mlp_model)], voting='soft')
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
    # model = XGBClassifier(n_estimators=1000, random_state=11)
    # F1Score -> 0.6444679654051167
    # model = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3, random_state=11)
    # F1Score -> 0.6311277078133634
    # model = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=128, random_state=11)

    # model = LGBMClassifier(n_estimators=1000)
    # 0.6428847951504623
    # model = LGBMClassifier(n_estimators=1000, max_depth=128, min_child_samples=100, num_leaves=64, subsample=0.8)
    # 0.6506658526347923

    model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 31, max_depth=-1, learning_rate = 0.1,
                           n_estimators = 40000, subsample_for_bin = 100000, objective = None, class_weight = 'balanced',
                           min_split_gain = 0.0, min_child_weight = 0.001, min_child_samples = 150, subsample = 1.0,
                           subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
                           random_state = None, n_jobs = - 1, importance_type = 'split')
    # 0.6518
    # model = LGBMClassifier(n_estimators=500, max_depth=128, min_child_samples=100, num_leaves=64, subsample=0.8)
    # 0.6493146786815271

    # model.fit(x_train, y_train)
    evals = [(x_test, y_test)]
    model.fit(x_train, y_train, early_stopping_rounds=100, eval_metric="f1", eval_set=evals, verbose=True)

    preds = model.predict(x_test)
    # pred_proba = model.predict_proba(x_test)[:, 1]

    # 파라미터 튜닝
    # GridSearchCV
    from sklearn.model_selection import GridSearchCV
    # from sklearn.metrics import make_scorer
    # f1 = make_scorer(f1_score, average='macro')
    # model = LGBMClassifier(n_estimators=200)
    #
    # params = {'num_leaves': [32, 64],
    #           'max_depth': [128, 160],
    #           'min_child_samples': [60, 100],
    #           'subsample': [0.8, 1]
    #           }

    # gridcv = GridSearchCV(model, param_grid=params, cv=3, scoring=f1)
    # gridcv.fit(x_train, y_train, early_stopping_rounds=30, eval_metric="f1", eval_set=[(x_train, y_train), (x_test, y_test)])
    #
    # print('best -> ', gridcv.best_params_)

    # visualize
    # from lightgbm import plot_importance
    # import matplotlib.pyplot asplt

    # fig, ax = plt.subplots(figsize=(10, 12))
    # plot_importance(model, ax=ax)

    # 평가   -> F1 Score
    from sklearn.metrics import f1_score

    f1 = f1_score(y_test, preds)

    print(f1)

    # # 저장
    # submission = pd.read_csv('sample_submission.csv')
    # submission['target'] = preds
    #
    # submission.to_csv('now.csv', index=False)
    #



