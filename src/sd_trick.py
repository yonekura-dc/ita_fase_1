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

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor

from data_utils import *
from scipy import stats
# -

import matplotlib.style as style

style.use("seaborn-notebook")
sns.set_palette("viridis")

df = pd.read_csv("../data/f_train.csv")
df_test = pd.read_csv("../data/f_test.csv")
df_trick = pd.read_csv("../data/sd_trick.csv")
df = df[df.cent_trans_cor > -0.38]

df_trick.columns.unique()

# +
mask = ['p', 'f', 'a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2',
    'c3', 'c4', 'g1', 'g2', 'l1', 'l2', 'l3', 'l4', 'e1', 'e2']

X = df_trick.drop(["sd_trans"],axis=1).values
y = df_trick["sd_trans"].values
model = LGBMRegressor(max_bin=255, lambda_l1=0.01, lambda_l2=0.1, learning_rate=.1, num_leaves=30, bagging_freq=1, bagging_fraction=.9, max_depth=12, objective='gamma')
model.fit(X, y)
# -

df['sd_trans'] = model.predict(df[mask])
df_test['sd_trans'] = model.predict(df_test[mask])

df = shuffle(df)
df.to_csv("../data/f_train.csv", index=False)
df_test.to_csv("../data/f_test.csv", index=False)

df.describe()

# +
plot_var_histogram = True
plot_var_cor = True

if plot_var_histogram:
    fig, ax = plt.subplots(figsize=(35, 30))
    for i, col in enumerate(df.columns):
        plt.subplot(7, 6, i + 1)
        sns.histplot(data=df, x=col, bins=60, kde=True, palette="viridis")

if plot_var_cor:
    fig, ax = plt.subplots(figsize=(30, 30))
    for i, col in enumerate(df.columns):
        plt.subplot(7, 6, i + 1)
        sns.scatterplot(data=df, x=col, y="cent_price_cor", palette="viridis")
# -

corrmat = df.corr(method="spearman")

f, ax = plt.subplots(figsize=(12, 8))
k = 36  # number of variables for heatmap
cols = corrmat.abs().nlargest(k, "cent_trans_cor")["cent_trans_cor"].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cmap="viridis",
    cbar=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)


