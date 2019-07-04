import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_dataset(fname):
    data = pd.read_csv(fname, encoding="utf-8")
    # drop掉无用数据
    data.drop(['obs_time', 'patient_id', 'dataset_name'], axis=1, inplace=True)
    return data


train = read_dataset('190624_data.csv')
y = train['is_vte_bool'].values
X = train.drop(['is_vte_bool'], axis=1).values
# y = y + 0  # 将布尔值替换成01

# 数据缺失值处理，采用众数策略
# strategy表示采用何种策略，有mean，median， most_frequent
# axis=0, 表示在列上面进行操作， axis=1表示在行上面进行操作
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(X)
X = imp.transform(X)

# 数据归一化
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X = min_max_scaler.fit_transform(X)
cols = train.columns

# 把数据的80%用来训练模型,20%做模型测试和评估,此处用到训练集-验证集二划分
p = 0.8  # 设置训练数据比例,
X_train = X[:int(len(X) * p), :]  # 前80%为训练集
X_test = X[int(len(X) * p):, :]  # 后20%为测试集
y_train = y[:int(len(y) * p)]  # 训练集标签
y_test = y[int(len(y) * p):]  # 测试集标签

# # 拆分训练集和测试集
# seed = 7
# test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size, random_state=seed)

float_columns = []
cat_columns = []
int_columns = []
for i in train.columns:
    if train[i].dtype == 'float':
        float_columns.append(i)
    elif train[i].dtype == 'int64':
        int_columns.append(i)
    elif train[i].dtype == 'object':
        cat_columns.append(i)
train_cat_features = train[cat_columns]
train_float_features = train[float_columns]
train_int_features = train[int_columns]
feature_names_cat = list(train_cat_features)
feature_names_float = list(train_float_features)
feature_names_int = list(train_int_features)
feature_names = sum([feature_names_cat, feature_names_float, feature_names_int], [])

# 拟合XGBoost模型
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# class_names = ['FALSE', 'TRUE']
# 评估预测结果
test_auc = metrics.roc_auc_score(y_test, y_pred)  # 验证集上的auc值
print(test_auc)

import lime.lime_tabular
import lime

predict_fn_xgb = lambda x: model.predict_proba(x).astype(float)
# Create the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['True', 'False'],
                                                   categorical_features=cat_columns,
                                                   categorical_names=feature_names_cat, kernel_width=3)

observation_1 = 2
exp = explainer.explain_instance(X_test[observation_1], predict_fn_xgb, num_features=6)

exp.show_in_notebook(show_all=False)
print(y_test[observation_1])
