import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Create dataframes for training and validation
from data_preprocess import combine

list_of_folders = ['CAT','CHMOD','CUT','DF','DU','ECHO','EXPAND','EXPR','FMT','HEAD','ID','KILL','LN','LS','NL','PATHCHK',
                    'PINKY','PR','PRINTF','PTX','SEQ','STAT','SUM','TAIL','TEST','TOUCH','UNEXPAND','UNIQ','WC','WHO']

data = combine()

data = data.astype(float)

X = data.drop(columns=['AFL_data','KLEE_data','taintCount_mean','taintCount_min','taintCount_max','taintCount_median'])
y_AFL = data['AFL_data']
y_KLEE = data['KLEE_data']

#Instantiate Bayesian Optimisation of XGBoost
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

ITERATIONS = 10

bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='exact'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')

    },    
    scoring = 'accuracy',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

#optimising on AFL data
bayes_cv_tuner.fit(X, y_AFL)
print('Best Estimator for AFL is : {}' .format(bayes_cv_tuner.best_estimator_))
print('Best Cross Validation Score on AFL data is : {}' .format(bayes_cv_tuner.best_score_))

clf_AFL = bayes_cv_tuner.best_estimator_

#optimising on KLEE data
bayes_cv_tuner.fit(X, y_KLEE)
print("Best Estimator for KLEE is : {}" .format(bayes_cv_tuner.best_estimator_))
print("Best Cross Validation Score on KLEE data is : {}" .format(bayes_cv_tuner.best_score_))

clf_KLEE = bayes_cv_tuner.best_estimator_

xgb.plot_importance(clf_AFL)
plt.title('Feature Importance for AFL')
plt.show()
xgb.plot_importance(clf_KLEE)
plt.title('Feature Importance for KLEE')
plt.show()

#saving the trained models as pickle files
""" import pickle

file_path = 'AFL_Model.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_AFL,model_pickle)
model_pickle.close()

file_path = 'KLEE_Model.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_KLEE,model_pickle)
model_pickle.close() """