import os
import csv
import pandas as pd


class StateLogger():

    def __init__(self, csv_path) -> None:
        assert csv_path.endswith('.csv'), 'Invalid file type, need CSV'

        self.csv_path = csv_path

    
    def log(self, state_dict):
        append_dict2csv(state_dict, self.csv_path)


    def check_exist(self, state_dict):
        '''Check row exists with condition specified in `state_dict`. 
        Only the 1st condition is considered now.
        '''
        # Future: support multiple condiiton through filterinfDataframe = dfObj[(dfObj['Sale'] > 30) & (dfObj['Sale'] < 33) ]
        if not os.path.exists(self.csv_path):
            return False

        #assert len(state_dict) == 1
        #key_str = list(state_dict.keys())[0]
        #val_str = state_dict[key_str]

        df = pd.read_csv(self.csv_path)
        #rows_filtered = df[df[key_str] == val_str]
        rows_filtered = df.loc[(df[list(state_dict)] == pd.Series(state_dict)).all(axis=1)]

        nrow_filtered = rows_filtered.shape[0]
        if nrow_filtered > 1:
            print('Warning: there exists multiple entries for %s where state is: %s'%(self.csv_path, str(state_dict)))

        if nrow_filtered == 0:
            return False
        else: # >0
            return True

def append_df2csv(df, csv_path):

    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path))

def append_dict2csv(data_dict, csv_path, insert_na=True):
    '''Append a data dict as a new row in specified csv file.
    If the file does not exist, create new file, and insert headers tgr with data_dict.
    If the file exists, directly append'''

    headers  = list(data_dict.keys()) 

    # write headers when creating the csv file
    if not os.path.exists(csv_path):  
        with open(csv_path, 'a+', newline='') as f:
            f_writer = csv.writer(f)
            f_writer.writerow(headers)
    else: # align columns
        df = pd.read_csv(csv_path)
        dict_keys = list(data_dict.keys())
        df_columns = list(df.columns)
        if dict_keys != df_columns:
            new_dict = {}
            for col in df_columns:
                if col in dict_keys:
                    new_dict[col] = data_dict[col]
                elif insert_na:
                    new_dict[col] = None
                    
            data_dict = new_dict
        

    # append data
    with open(csv_path, 'a', newline='') as f:
        dictwriter = csv.DictWriter(f, fieldnames=headers)
        dictwriter.writerow(data_dict)


if __name__ == '__main__':
    logger = StateLogger('./test.csv')
    state_dict = {'a':1, 'b':2}
    logger.log(state_dict)
    state_dict['b']=3
    logger.log(state_dict)
    state_dict['b']=1
    logger.log(state_dict)
    print(logger.check_exist(state_dict))
    state_dict['a'] = 2
    print(logger.check_exist(state_dict))
    