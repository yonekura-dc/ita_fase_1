import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.kernel_ridge import KernelRidge

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.metrics import make_scorer


def mae_vectorized(a, b):
    return np.mean(np.abs(a - b))


def hellinger_distance(p, q):
    p = np.sqrt(np.sort(p) / np.max(p))
    q = np.sqrt(np.sort(q) / np.max(q))
    return np.sum(np.square(p - q) / np.sqrt(2))
