import pandas as pd
import numpy as np
import sys

from data_preprocess import data_filter

path_to_file = sys.argv[1]

features = pd.DataFrame(columns = ['trace_length','global_nesting_depth','call_depth','no_of_local_variables','taintCount'])
data = data_filter(path_to_file)
features = features.append(data,ignore_index=True)
print(features)

import pickle

model_path = 'AFL_Model_DT.pkl'
model_unpickle = open(model_path,'rb')
clf_AFL = pickle.load(model_unpickle)

model_path = 'KLEE_Model_DT.pkl'
model_unpickle = open(model_path,'rb')
clf_KLEE = pickle.load(model_unpickle)


pred_AFL = clf_AFL.predict(features)
pred_KLEE = clf_KLEE.predict(features)

print('Prediction for AFL is {}'.format(pred_AFL))
print('Prediction for KLEE is {}'.format(pred_KLEE)) 