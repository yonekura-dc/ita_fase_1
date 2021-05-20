import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'cent_price_cor' in the data file
tpot_data = pd.read_csv('../../data/f_train.csv')

X = tpot_data.drop(["cent_price_cor", "cent_trans_cor"], axis=1).values
y = tpot_data[["cent_price_cor"]].values

kf = KFold(n_splits=10)
scores = []

# Average CV score on the training set was: 0.09377849321488732
exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.35000000000000003),
    GradientBoostingRegressor(alpha=0.8, learning_rate=0.001, loss="lad", max_depth=8, max_features=0.6000000000000001, min_samples_leaf=15, min_samples_split=18, n_estimators=100, subsample=0.05)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    exported_pipeline.fit(X_train, y_train)
    y_pred = exported_pipeline.predict(X_test)
    print("mae: ", mean_absolute_error(y_test, y_pred))
    scores.append(mean_absolute_error(y_test, y_pred))

print("media:", np.mean(scores))
