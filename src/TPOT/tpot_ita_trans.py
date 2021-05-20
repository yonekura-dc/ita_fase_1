import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'cent_trans_cor' in the data file
tpot_data = pd.read_csv('../../data/f_train.csv')
X = tpot_data.drop(["cent_price_cor", "cent_trans_cor"], axis=1).values
y = tpot_data[["cent_trans_cor"]].values

kf = KFold(n_splits=10)
scores = []

# Average CV score on the training set was: 0.09005241503770736
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    Binarizer(threshold=0.4),
    GradientBoostingRegressor(alpha=0.75, learning_rate=0.001, loss="lad", max_depth=9, max_features=0.8, min_samples_leaf=20, min_samples_split=3, n_estimators=100, subsample=1.0)
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