import pandas as pd
import numpy as np

data = pd.read_csv('main.csv')

X = data.drop(columns=['AFL_data','KLEE_data'])

X = X.drop(columns=['trace_length_mean','global_nesting_depth_mean','call_depth_mean'])

y_AFL = data['AFL_data']
y_KLEE = data['KLEE_data']

for column_name in X.columns:
    new_column = str(column_name)+'_neg'
    X[new_column] = X[column_name].apply(lambda val: val<X[column_name].median())
    X[new_column] = X[new_column].map({True:1,False:0})
    print('{} : {}'.format(column_name,X[column_name].median()))
    X[column_name] = X[column_name].apply(lambda val: val>X[column_name].median())
    X[column_name] = X[column_name].map({True:1,False:0})

y_AFL = y_AFL.map({1:0,0:1})
y_KLEE = y_KLEE.map({1:0,0:1})

X = pd.concat([X,y_AFL],axis=1,join='inner')

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(X, min_support=0.05, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

rules = rules[(rules['consequents'] == frozenset({'AFL_data'}))]

rules = rules.sort_values(by=['lift'],ascending=False)

rules.to_csv('Assocition_Rules_median_threshold_afl_false.csv')