import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz

from data_preprocess import combine

list_of_folders = ['CAT','CHMOD','CUT','DF','DU','ECHO','EXPAND','EXPR','FMT','HEAD','ID','KILL','LN','LS','NL','PATHCHK',
                    'PINKY','PR','PRINTF','PTX','SEQ','STAT','SUM','TAIL','TEST','TOUCH','UNEXPAND','UNIQ','WC','WHO']

data = combine()

data = data.astype(float)

X = data.drop(columns=['AFL_data','KLEE_data','taintCount_mean','taintCount_min','taintCount_max','taintCount_median','trace_length_mean','trace_length_min','trace_length_median',
                        'global_nesting_depth_max','global_nesting_depth_min','global_nesting_depth_median','call_depth_median','call_depth_min','call_depth_max',
                        'no_of_local_variables_mean','no_of_local_variables_min','no_of_local_variables_median'])
y_AFL = data['AFL_data']
y_KLEE = data['KLEE_data']

from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

ITERATIONS = 25

bayes_cv_tuner = BayesSearchCV(
    estimator = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        presort=True
    ),  
    search_spaces = {
        'max_features': (1,4),
        'max_depth':(1,6),
        'min_samples_split': (2,10),
        'min_samples_leaf': (1,10),
        'min_weight_fraction_leaf': (1e-9,0.5,'log-uniform'),
        'min_impurity_decrease' : (1e-9,0.5,'log-uniform'),
        'max_leaf_nodes':(2,30)
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

import graphviz

export_graphviz(clf_AFL, out_file='AFL_tree.dot',   
                         filled=True, rounded=True,  
                         special_characters=True,
                         class_names=['0','1'],feature_names=['trace_length_max','no_of_local_variables_max','global_nesting_depth_mean','call_depth_mean'])  

export_graphviz(clf_KLEE, out_file='KLEE_tree.dot',   
                         filled=True, rounded=True,  
                         special_characters=True,
                         class_names=['0','1'],feature_names=['trace_length_max','no_of_local_variables_max','global_nesting_depth_mean','call_depth_mean'])  

"""#saving the trained models as pickle files
import pickle

file_path = 'AFL_Model_combined.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_AFL,model_pickle)
model_pickle.close()

file_path = 'KLEE_Model_combined.pkl'
model_pickle = open(file_path, 'wb')
pickle.dump(clf_KLEE,model_pickle)
model_pickle.close()"""

