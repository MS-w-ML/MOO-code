import sys
import sqlite3
import pandas as pd


class Database():
    '''Interface between pandas.DataFrame and sqlite3 database.'''
    def __init__(self, db_path):
        self.db_path = db_path

        assert db_path.endswith('.db'), 'Invalid database path: %s: it shall ends with \'.db\''% db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        # create table to store data info 
        data_info_sql = 'CREATE TABLE IF NOT EXISTS data_info (id, save_pointer, read_pointer); INSERTINTO data_info VALUES (0,1,1)'
        self.cursor.execute(data_info_sql)
        
    def __len__(self):
        data_info_sql = 'SELECT save_pointer FROM data_info WHERE id=0;'
        self.cursor.execute(data_info_sql)
        save_data_ptr = self.cursor.fetchall()[0]
        return save_data_ptr - 1

    def __getitem__(self, index):
        # load data
        sql_str = 'SELECT * FROM data WHERE id=%d ;'%(index)
        df = pd.read_sql_query(sql_str, self.connection)
        return df

    def getbatch(self, indexes):
        index_str ='('#('Germany', 'France', 'UK')
        for ind in indexes:
            index_str += str(ind) + ','
        # load data
        index_str  = index_str[:-1]
        index_str += ')'
        sql_str = 'SELECT * FROM data WHERE id IN %s;'%(index_str)
        df = pd.read_sql_query(sql_str, self.connection)
        return df 

    def read_chunk(self, chunk_size=1e6):
        # read data info and get data indexes to read
        data_info_sql = 'SELECT save_pointer, read_pointer FROM data_info WHERE id=0;'
        self.cursor.execute(data_info_sql)
        save_data_ptr, read_data_ptr = self.cursor.fetchall()[0]

        if save_data_ptr > read_data_ptr:
            raise RuntimeError('Empty table or Read pointer exceeds table size: read pointer=%d and save pointer=%d'%(read_data_ptr, save_data_ptr))
        max_index = min(read_data_ptr+chunk_size, save_data_ptr)

        # load data
        sql_str = 'SELECT * FROM data WHERE id>=%d AND id<%d ;'%(read_data_ptr, max_index)
        df = pd.read_sql_query(sql_str, self.connection)
        df.set_index('id')

        # update data info table
        data_info_sql = 'UPDATE data_info SET read_pointer=%d WHERE id=0;'%max_index
        self.cursor.execute(data_info_sql)
        self.connection.commit()
        return df

    def save_chunk(self, df):
        '''Save pandas.DataFrame object into database. Update index
        '''
        # read data info 
        data_info_sql = 'SELECT save_pointer FROM data_info WHERE id=0;'
        self.cursor.execute(data_info_sql)
        save_data_ptr = self.cursor.fetchall()[0][0]

        # update indexes and other values
        df['id'] = range(save_data_ptr, save_data_ptr+df.shape[0])
        df.set_index('id')
        save_data_ptr += df.shape[0]
        
        # udpate data table
        df.to_sql(name='data', con=self.connection, if_exists='append', index=False)
        
        # update data info table
        data_info_sql = 'UPDATE data_info SET save_pointer=%d WHERE id=0;'%save_data_ptr
        self.cursor.execute(data_info_sql)
        self.connection.commit()

    def reset_reader(self):
        # update data info table
        data_info_sql = 'UPDATE data_info SET read_pointer=1 WHERE id=0;'
        self.cursor.execute(data_info_sql)
        self.connection.commit()

    def read_iterator(self, chunk_size=1e6, reset_reader=True):
        while True:
            try:
                yield self.read_chunk(chunk_size)
            except:
                print("Runtime",sys.exc_info()[0],"occured.")
                if reset_reader: 
                    self.reset_reader()
                break


    def close_connection(self):
        self.connection.commit()
        self.connection.close()