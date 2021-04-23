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

# +
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import IsolationForest


from data_utils import *
import xgboost as xgb

# -

df = pd.read_csv("../data/f_train.csv")
X = df.drop(["cent_price_cor", "cent_trans_cor"], axis=1)
y = df["cent_trans_cor"].values

model = RandomForestRegressor(
    n_estimators=200, max_depth=15, random_state=1337, n_jobs=4, min_samples_leaf=2
)

model.fit(X, y)

importances = model.feature_importances_
final_df = pd.DataFrame({"Features": X.columns, "Importances": importances})

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=final_df, x="Features", y="Importances", palette="viridis")
plt.xticks(rotation=45)
plt.show()

"""
estimator = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=1337, n_jobs=4, min_samples_leaf=2)
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.ranking_
"""

out_detector = IsolationForest(contamination=0.01)
out_detector.fit(df)
isolated = pd.DataFrame(out_detector.predict(df), columns=["out"])
isolated[isolated.out == -1] = 0
df_filtered = df[isolated.out == 1]

df_filtered.to_csv("../data/outlier_removed.csv", index=False)
