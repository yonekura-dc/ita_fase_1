# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: ita
#     language: python
#     name: ita
# ---

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import scipy

# +
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.neural_network import MLPRegressor
from sklearn.tree import ExtraTreeRegressor

import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error
from data_utils import *


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_validate


# +
df = pd.read_csv("../data/f_train.csv")
bruh = ["n", "b1", "b3", "g1", "l1", "l3", "e_avg", "g_l", "rf1", "volume", "floor_area", "rf3", "sd_trans", "cent_price_cor", "cent_trans_cor"]
#X = df[bruh].values
X = df[bruh].drop(["cent_price_cor", "cent_trans_cor"], axis=1).values

inputs = ["cent_price_cor", "cent_trans_cor"]  # Use this for multi-output models
inputs = ["cent_trans_cor"]  # Use this for single output models
y = df[inputs].values

# -


def plot_pred_ensemble(model, X):
    preds = pd.DataFrame()
    for name, pred in model.named_estimators_.items():
        preds[name] = pred.predict(X)
    preds["ensemble"] = model.predict(X)
    preds.sort_values(by=["ensemble"], inplace=True)
    preds = preds.apply(np.square)
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for i in preds.columns:
        sns.lineplot(data=preds, x=np.arange(len(X)), y=i, label=i)
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.histplot(data=preds, x="ensemble", bins=10)
    plt.show()


# +
#model = MLPRegressor(4, activation="relu", solver="adam", max_iter=5000, alpha=.1, max_fun=20000)
#model = SGDRegressor(alpha=.00000)

# model = NuSVR(nu=0.4, C=300, gamma="auto")
#model = KernelRidge(alpha=1, degree=2, kernel="poly")#, gamma=.05)

#model = RandomForestRegressor(n_estimators=30, max_depth=2, random_state=1337, n_jobs=4)

#model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=2, reg_alpha=0.1, reg_lambda=0.1, gamma=0.3, subsample=0.9,min_child_weight=0.5, n_jobs=4)
#model = CatBoostRegressor(verbose=False)
#model = LGBMRegressor(max_bin=255, lambda_l1=0.1, lambda_l2=0.1, learning_rate=.1, num_leaves=20, bagging_freq=1, bagging_fraction=.9, max_depth=15, verbose=0)

#model = BayesianRidge()
#model = Ridge(0.1)
estimators = [  #("kernel_ridge", KernelRidge(alpha=1, degree=2, kernel="poly", gamma=.05)),
#     (
#         "xgb",
#         xgb.XGBRegressor(
#             n_estimators=1000,
#             learning_rate=0.1,
#             max_depth=3,
#             reg_alpha=0.1,
#             reg_lambda=0.1,
#             gamma=0.3,
#             subsample=0.9,
#             min_child_weight=0.5,
#         ),
#     ),
    #("ann", MLPRegressor(8, solver="lbfgs", max_iter=5000, alpha=0.5, learning_rate_init=0.002, max_fun=20000)),
    (
        "lgbm",
        LGBMRegressor(
            max_bin=255,
            lambda_l1=0.01,
            lambda_l2=0.1,
            learning_rate=0.1,
            num_leaves=20,
            bagging_freq=1,
            bagging_fraction=0.9,
            max_depth=12,
        ),
    ),
    ("cat", CatBoostRegressor(verbose=False)),
    ("b_ridge", BayesianRidge()),
    ("ridge", Ridge(0.1)),
    ("lasso", Lasso())
    # ("svr", NuSVR(nu=0.4, C=300, gamma="auto"))
]
model = VotingRegressor(estimators=estimators)#, weights=[0.4, 0.4, 0.2])
#model = Lasso()
# model = StackingRegressor(estimators=estimators, n_jobs=4)
# model = GradientBoostingRegressor(n_estimators=600, max_depth=2, random_state=1337)


# +
k = 10
kf = KFold(n_splits=k, shuffle=True)
scores = []
cheats = []



for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("mae: ", mean_absolute_error(y_test, y_pred))
    scores.append(mean_absolute_error(y_test, y_pred))

    
    plot_pred_ensemble(model, X_test) # enable this if running ensembles

print("media:", np.mean(scores))
# -

model.fit(X, y)
y_pred = model.predict(X)
print("mae: ", mean_absolute_error(y, y_pred))


df_out = pd.DataFrame(model.predict(X_test), columns=inputs)
df_out.describe()

# +
sns.histplot(data=df, x="cent_price_cor", stat="probability", bins=30, color="black")
sns.histplot(data=df, x="cent_trans_cor", stat="probability", bins=30, color="w")

for i in inputs:
    sns.histplot(data=df_out, x=i, stat="probability", bins=30, color="r")
# -


