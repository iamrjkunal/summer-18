import pandas as pd
import numpy as np
import sys

import pickle
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

""" from data_preprocess import combine

list_of_folders = ['CAT','CHMOD','CUT','DF','DU','ECHO','EXPAND','EXPR','FMT','HEAD','ID','KILL','LN','LS','NL','PATHCHK',
                    'PINKY','PR','PRINTF','PTX','SEQ','STAT','SUM','TAIL','TEST','TOUCH','UNEXPAND','UNIQ','WC','WHO']

data = combine(list_of_folders)

data = data.astype(float)
data = data.drop(columns=['AFL_data','KLEE_data'])

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
 """
model_path = 'AFL_Model_combined.pkl'
model_unpickle = open(model_path,'rb')
dtc_AFL = pickle.load(model_unpickle)

model_path = 'KLEE_Model_combined.pkl'
model_unpickle = open(model_path,'rb')
dtc_KLEE = pickle.load(model_unpickle)

""" 
afl_plots = plot_partial_dependence(dtc_AFL,       
                                   features=[0,1,2,3,4],
                                   X=data,
                                   feature_names=['trace_length', 'global_nesting_depth',  'call_depth', 'no_of_local_variables', 'taintcount'])
                                       
plt.show()


klee_plots = plot_partial_dependence(dtc_KLEE,       
                                   features=[0,1,2,3,4],
                                   X=data,
                                   feature_names=['trace_length', 'global_nesting_depth',  'call_depth', 'no_of_local_variables', 'taintcount'])
                                
plt.show() """
import graphviz

export_graphviz(dtc_AFL, out_file='AFL_tree.dot',   
                         filled=True, rounded=True,  
                         special_characters=True,
                         class_names=['0','1'],feature_names=['trace_length_max','global_nesting_depth_max','call_depth_max',
                         'no_of_local_variables_max','taintCount_max','trace_length_mean','global_nesting_depth_mean','call_depth_mean',
                         'no_of_local_variables_mean','taintCount_mean','trace_length_median','global_nesting_depth_median','call_depth_median',
                         'no_of_local_variables_median','taintCount_median'])  

export_graphviz(dtc_KLEE, out_file='KLEE_tree.dot',   
                         filled=True, rounded=True,  
                         special_characters=True,
                         class_names=['0','1'],feature_names=['trace_length_max','global_nesting_depth_max','call_depth_max',
                         'no_of_local_variables_max','taintCount_max','trace_length_mean','global_nesting_depth_mean','call_depth_mean',
                         'no_of_local_variables_mean','taintCount_mean','trace_length_median','global_nesting_depth_median','call_depth_median',
                         'no_of_local_variables_median','taintCount_median'])  
"""['trace_length_max','global_nesting_depth_max','call_depth_max',
                         'no_of_local_variables_max','taintCount_max','trace_length_mean','global_nesting_depth_mean','call_depth_mean',
                         'no_of_local_variables_mean','taintCount_mean','trace_length_median','global_nesting_depth_median','call_depth_median',
                         'no_of_local_variables_median','taintCount_median','trace_length_min','global_nesting_depth_min','call_depth_min',
                         'no_of_local_variables_min','taintCount_min'] """