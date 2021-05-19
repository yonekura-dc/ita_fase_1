from tpot import TPOTRegressor
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/f_train.csv")
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


df_test = pd.read_csv("../../data/f_test.csv")
test = df_test.drop(["id"], axis=1)
id_test = df_test["id"]

y_pred1 = model1.predict(test)
y_pred2 = model2.predict(test)

df_out = {"cent_price_cor": y_pred1.reshape(-1), "cent_trans_cor": y_pred2.reshape(-1)}
result = pd.DataFrame(df_out)

result.to_csv('result.csv', index=False)
