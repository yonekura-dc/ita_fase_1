# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from autogluon.tabular import TabularDataset, TabularPredictor
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import dirname
from sklearn.model_selection import KFold

OUT_DIR = dirname(__file__)+'/out2'
DATA_DIR = '/app/data'

# data = TabularDataset(DATA_DIR+'/f_train.csv')
# data = data.sample(n=500, random_state=42)
# data.head()

# # + tags=[] jupyter={"source_hidden": true, "outputs_hidden": true}
# label = "cent_price_cor"
# #label = "cent_trans_cor"'
# label_to_drop = "cent_trans_cor" if label == "cent_price_cor" else "cent_price_cor"
# data = data.drop(columns=[label_to_drop])


# + tags=[] jupyter={"source_hidden": true}
def run_feature_importance(label, test_data, predictor, path_output):
    feature_importance = predictor.feature_importance(test_data)
    with open(path_output, 'a+') as f:
        content = ("-- feature importance --\n" +
                   "{}\n").format(feature_importance.to_csv(sep='\t'))
        f.write(content)


# + tags=[]
def run_plot(label, test_data_nolabel, predictor, data):
    df_out = pd.DataFrame(predictor.predict(test_data_nolabel))
    
    sns.histplot(data=data, x=label, stat="probability", bins=30, color="black")
    sns.histplot(data=df_out, x=label, stat="probability", bins=30, color="r")
    plt.savefig(OUT_DIR+'/output-{}.png'.format(label))
    plt.clf()


# + tags=[]
def run(label, data):

    kf = KFold(n_splits=2, random_state=42, shuffle=True)

    soma = 0
    valores = []
    n_splits = kf.get_n_splits(data)
    for fold, n_split in zip(kf.split(data), range(n_splits)):
        train_data = data.iloc[fold[0]]
        test_data = data.iloc[fold[1]]

        save_path = 'agModels-{}'.format(label)  # specifies folder to store trained models
        metric = 'mae'

        predictor = TabularPredictor(label=label, path=save_path,problem_type='regression', 
                                     eval_metric=metric).fit(train_data, presets='best_quality')

        y_test = test_data[label]  # values to predict
        test_data_nolabel = test_data.drop(columns=[label])

        y_pred = predictor.predict(test_data_nolabel)

        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        score = perf['mean_absolute_error']
        print('Resultado MAE (split {}):'.format(n_split), score)
        valores.append(score)
        soma += score

    mae_media = soma/n_splits
    print('Resultado MAE (média):', mae_media)

    path_output = OUT_DIR+'/resultados_{}.txt'.format(label)

    with open(path_output, 'a+') as f:
        content = ("----------------------------------\n" +
                   "Timestamp: {}\n" +
                   "MAE média: {}\n" +
                   "MAE por split: {}\n").format(datetime.now(), str(mae_media), [str(valor)+' ' for valor in valores])
        f.write(content)
    
    run_feature_importance(label, test_data, predictor, path_output)
    run_plot(label, test_data_nolabel, predictor, data)


# + tags=[]
labels = ["cent_price_cor", "cent_trans_cor"]
for label in labels:
    data = TabularDataset(DATA_DIR+'/f_train.csv')
    data = data.sample(n=500, random_state=42)
    label_to_drop = "cent_trans_cor" if label == "cent_price_cor" else "cent_price_cor"
    data = data.drop(columns=[label_to_drop])
    run(label, data)
