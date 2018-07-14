import pandas as pd

#function for extracting data from a file
def data_filter(path_to_file):
    raw_data = pd.read_csv(path_to_file,names=['line_number','file_name','line_id','trace_length','global_nesting_depth',
                                                'call_depth','no_of_local_variables','taintCount','text'])
    raw_data = raw_data.drop(['line_number','file_name','line_id','text'],axis=1)
    df = raw_data.max(axis=0)
    return df

#function for iterating over all folders
def data_extractor(folder_list):
    df = pd.DataFrame(columns = ['trace_length','global_nesting_depth','call_depth','no_of_local_variables','taintCount'])
    for folder_name in folder_list:
        for subfolder_name in ['1','2','3','4']:
            path = 'tosent/'+ folder_name + '/' + subfolder_name + '/AvailableVariablesFiltered.txt'
            features = data_filter(path)
            df = df.append(features,ignore_index=True)
    return df

#functions for extracting target data
def target_data_extractor():
    AFL_df = pd.read_csv('AFLSingle.csv',names=['folder_name1','subfolder_name','AFL_data','useless_data1',
                        'useless_data2','useless_data3','useless_data4','useless_data5'],delimiter=' ')
    AFL_df = AFL_df.sort_values(by=['folder_name1','subfolder_name'])
    AFL_df = AFL_df.drop(['folder_name1','subfolder_name','useless_data1',
                       'useless_data2','useless_data3','useless_data4','useless_data5'],axis=1)
    AFL_df = AFL_df.reset_index(drop=True)

    KLEE_df = pd.read_csv('KLEESingle.csv',names=['folder_name2','subfolder_name','KLEE_data','useless_data1',
                        'useless_data2','useless_data3','useless_data4'],delimiter=' ')
    KLEE_df = KLEE_df.sort_values(by=['folder_name2','subfolder_name'])
    KLEE_df = KLEE_df.drop(['folder_name2','subfolder_name','useless_data1',
                       'useless_data2','useless_data3','useless_data4'],axis=1)
    KLEE_df = KLEE_df.reset_index(drop=True)

    target_data = pd.concat([AFL_df, KLEE_df],axis=1,join='inner')
    target_data['AFL_data'] = target_data['AFL_data'].apply(lambda val: val>0)
    target_data['KLEE_data'] = target_data['KLEE_data'].apply(lambda val: val>0)
    target_data['AFL_data'] = target_data['AFL_data'].map({True:1,False:0})
    target_data['KLEE_data'] = target_data['KLEE_data'].map({True:1,False:0})

    return target_data

#function for combining features and target data
def combine():
    data = pd.read_csv('combined_data.csv')
    target = target_data_extractor()
    combine = pd.concat([data,target],axis=1,join='inner')
    return combine

