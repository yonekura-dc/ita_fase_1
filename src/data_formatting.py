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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from data_utils import *
from scipy import stats

import matplotlib.style as style

style.use("seaborn-notebook")
sns.set_palette("viridis")

df = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")

sns.histplot(data=df, x="cent_price_cor")
sns.histplot(data=df, x="cent_trans_cor")
df = df[df.cent_trans_cor > -0.38]


# +
def feature_generation(df):
    df["g1"] = df["g1"].apply(np.sqrt)
    df["g2"] = df["g2"].apply(np.sqrt)

    df["e_avg"] = df.filter(regex=("e\d")).sum(axis=1)

    df["g_l"] = df["g1"] * df["g2"] * df["l3"] * df["n"]

    df["a_1_3"] = df["a1"] - df["a3"]
    df["a_1_3"] += np.abs(df["a_1_3"].min()) + 1

    df["c_1_3"] = df["c1"] - df["c3"]
    df["c_1_3"] += np.abs(df["c_1_3"].min()) + 1

    df["rf1"] = df["g_l"] + df["a_1_3"]
    df["rf2"] = df["c_1_3"] - df["a_1_3"]
    df["rf2"] += np.abs(df["rf2"].min()) + 1

    # df['coord_1'] = (df['x'] + df['y'] + df['z'])/3
    # df['coord_2'] = (df['x'] * df['y'] * df['z'])**(1./3)
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["floor_area"] = df["x"] * df["y"]
    df["rf3"] = df["volume"] * (df["g1"] + 0.1)
    # df['density'] = df['volume'] / df['n']

    return df


# BoxCox
def normalize(df, df_test):
    for col in df.columns:
        if col not in ["cent_price_cor", "cent_trans_cor", "id"]:
            df[col], lmbda = stats.boxcox(df[col] + 0.1)
            df_test[col] = stats.boxcox(df_test[col] + 0.1, lmbda=lmbda)

            scaler = RobustScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            df[col] = scaler.transform(df[col].values.reshape(-1, 1))
            df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))
    return df, df_test


# MinMax Scaling
def scale(df, df_test):
    for col in df.columns:
        if col not in ["cent_price_cor", "cent_trans_cor", "id"]:
            scaler = MinMaxScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            df[col] = scaler.transform(df[col].values.reshape(-1, 1))
            df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))
    return df, df_test


df = feature_generation(df)
df_test = feature_generation(df_test)
# df, df_test = scale(df, df_test)
df, df_test = normalize(df, df_test)

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

# +
plot_var_histogram = False
plot_var_cor = False

if plot_var_histogram:
    fig, ax = plt.subplots(figsize=(35, 30))
    for i, col in enumerate(df.columns):
        plt.subplot(7, 6, i + 1)
        sns.histplot(data=df, x=col, bins=60, kde=True, palette="viridis")

if plot_var_cor:
    fig, ax = plt.subplots(figsize=(30, 30))
    for i, col in enumerate(df.columns):
        plt.subplot(7, 6, i + 1)
        sns.scatterplot(data=df, x=col, y="cent_trans_cor", palette="viridis")

# -

df = shuffle(df)
df.to_csv("../data/f_train.csv", index=False)
df_test.to_csv("../data/f_test.csv", index=False)

df.describe()
