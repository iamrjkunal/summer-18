import pandas as pd
import numpy as np

from data_preprocess import combine

list_of_folders = ['CAT','CHMOD','CUT','DF','DU','ECHO','EXPAND','EXPR','FMT','HEAD','ID','KILL','LN','LS','NL','PATHCHK',
                    'PINKY','PR','PRINTF','PTX','SEQ','STAT','SUM','TAIL','TEST','TOUCH','UNEXPAND','UNIQ','WC','WHO']

data = combine(list_of_folders)

X = data.drop(columns=['AFL_data','KLEE_data'])
y_AFL = data['AFL_data']
y_KLEE = data['KLEE_data']

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

ITERATIONS = 10

bayes_cv_tuner = BayesSearchCV(
    estimator = GradientBoostingClassifier(),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_features': (1,5),
        'max_depth':(1, 5),
        'min_samples_split': (2,10),
        'min_samples_leaf': (1,10),
        'min_weight_fraction_leaf': (1e-9,0.5,'log-uniform'),
        'max_leaf_nodes':(2,100),
        'n_estimators': (50,100)
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

#saving the trained models as pickle files
import pickle

file_path = 'AFL_Model_gbrt.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_AFL,model_pickle)
model_pickle.close()

file_path = 'KLEE_Model_gbrt.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_KLEE,model_pickle)
model_pickle.close()