from tpot import TPOTRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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

# +
df = pd.read_csv("../../data/train.csv")
df_test = pd.read_csv("../../data/test.csv")

df = df[df.cent_trans_cor > -0.38]
df = feature_generation(df)
df_test = feature_generation(df_test)
df, df_test = normalize(df, df_test)
# -

df.to_csv("../../data/f_train.csv", index=False)

X = df.drop(["cent_price_cor", "cent_trans_cor"], axis=1).values
Y1 = df[["cent_price_cor"]].values
Y2 = df[["cent_trans_cor"]].values

kf = KFold(n_splits=10)

model1 = TPOTRegressor(generations=10, population_size=100, cv=kf, scoring='neg_mean_absolute_error', verbosity=2, random_state=42)
print(model1.fit(X,Y1))
model1.export('tpot_ita_model1.py')

model2 = TPOTRegressor(generations=10, population_size=100, cv=kf, scoring='neg_mean_absolute_error', verbosity=2, random_state=42)
print(model2.fit(X,Y2))
model2.export('tpot_ita_model2.py')


df_test.to_csv("../../data/f_test.csv", index=False)
test = df_test.drop(["id"], axis=1)
id_test = df_test["id"]

y_pred1 = model1.predict(test)
y_pred2 = model2.predict(test)

df_out = {"cent_price_cor": y_pred1.reshape(-1), "cent_trans_cor": y_pred2.reshape(-1)}
result = pd.DataFrame(df_out)

result.to_csv('result.csv', index=False)
