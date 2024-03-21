################################################################
#   Helper function
################################################################
import os
import datetime
import shutil
import numpy as np
import pandas as pd
from utils.utils import normalize


def load_data(ds_name, from_dir='./data/', y_type=None, norm=False):
    '''
    Load csv file into dataframes.
    '''

    if(ds_name=='cqd_raw'):
        df = pd.read_csv(from_dir+'cqd_raw.csv')
        assert y_type == 'yield'

    elif(ds_name=='hydro_new'):
        df = pd.read_csv(from_dir+'hydro_new.csv')
        assert y_type == 'yield'
        
    elif(ds_name=='moo'):
        df = pd.read_csv(from_dir+'moo_63.csv')
        lbl_list = ['color', 'yield']
        if y_type is not None: # drop all Y columns
            if y_type in lbl_list:
                lbl_list.remove(y_type)
            else:
                raise Exception('Error: invalid y_type %s  for dataset %s specified.'%(y_type, ds_name))
            
        df = df.drop([i+'_real' for i in lbl_list], axis=1)
        if y_type == 'color':
            df = df[df['color_real'].notnull()] # filter none
            if norm:
                df['color_real'] = normalize(x=df['color_real'], old_max=max(df['color_real']), old_min=min(df['color_real']), new_max=1, new_min=0)
            else:
                df['color_real'] = df['color_real']

    else:
        raise Exception('Error: invalid dataset name specified: %s'%ds_name)


    return df
    
    
def load_XY(ds_name, from_dir='./data/', as_matrix=False, y_type=None, norm=False, reset_index=False):
    '''Load dataset into X,Y.
    '''

    df= load_data(ds_name, from_dir, y_type=y_type, norm=norm)
    
    feature_list = df.columns[0:len(df.columns)-1]
    result_col = df.columns[len(df.columns)-1]
    X = df[feature_list]
    Y = df[result_col]

    if y_type=='yield' and (Y>1).any(): # if yield is in range [0,100]
        Y = Y / 100

    if as_matrix:
        X = X.to_numpy()
        Y = Y.to_numpy()
    elif reset_index:
        X.reset_index(inplace=True, drop=True)
        Y.reset_index(inplace=True, drop=True)
    print('Loaded %s dataset...'%ds_name)
    return X,Y 
'''
def load_csv(title,isData=True):
    if(isData):
        result = pd.read_csv(from_dir+title+'.csv')
    else:
        result = pd.read_csv(to_dir+title+'.csv')
    return result
'''

def list_generator(start, end, step):
    list_ = list(range(int(start*100),int(end*100),int(step*100))) + [end*100]
    results = [float("{0:.2f}".format(x*0.01)) for x in list_]
    return(results)


def save_csv(data,to_save_title,ind=False):

    data.to_csv(to_save_title,index=ind)
    print('Successfully saved :',to_save_title)
    return to_save_title

def append_csv(filepath, data):
    '''
    params:
        filepath:   <os.path> Full csv path, e.g. ./a/b/name.csv
        row_dict:   <dict | pandas.Dataframe | numpy.array>. Rows to be append
    '''
    # convert dict to pandas 
    assert isinstance(data, dict), 'Invalid data type: %s. Only <dict> accepted'%type(data)
    data = pd.DataFrame.from_dict(data, orient='index').T

    if not os.path.exists(filepath): # write header
        data.to_csv(filepath,mode='a',index=False,header=True)
            #writer.writerows(data)
    else:
        data.to_csv(filepath,mode='a',index=False,header=False)


def save_csv_w_format_title(data, to_dir,  title, index=False, overwrite=False, verbose=False, format_date=False):
    '''
    params:
        data:   <pandas.Dataframe> or Dict
        to_dir: <os.path>
        title:  <string>
        index:  <Bool> Whether to save index of data into csv
        overwrite: <Bool>. Whether to overwrite file with exactly the same title, i.e. $title$.csv
        verbose: <Bool>
        format_date: <Bool>. Whether to include date in title.

    '''
    
    origin_title = os.path.join(to_dir, title+'.csv')

    if overwrite and os.path.isfile(origin_title):
        os.remove(origin_title)
    else:
        to_save_title = format_title(to_dir, title, fileEtd='.csv', format_date= format_date)

    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    data.to_csv(to_save_title,index=index)
    if verbose:
        print('Successfully saved :',to_save_title)
    return to_save_title


def update_title_w_date(title):
    now_time = datetime.datetime.now()
    today = str(now_time.year)+'_'+str(now_time.month)+'_'+str(now_time.day)
    return title + today

def format_title(to_dir, title, fileEtd, format_date):
    if format_date:
        title = update_title_w_date(title)
    to_save_title = os.path.join(to_dir, title+fileEtd)
    
    i=0
    while(os.path.exists(to_save_title)):
        to_save_title = os.path.join(to_dir,title+'_'+str(i)+fileEtd)
        i= i+1
    return to_save_title


def mkdirs(dirs,overwrite=False, ignore=False):
    for dir in dirs:
        mkdir(dir, overwrite, ignore=ignore)

def mkdir(dir, overwrite=False, ignore=False):
    if os.path.exists(dir):
        if overwrite:
            shutil.rmtree(dir)
            os.mkdir(dir)
        elif not ignore:
            raise UserWarning('Directory exists: %s'%str(dir))
        else:
            return
    else:
        os.mkdir(dir)

def list_generator(start, end, step):
    '''Generator list of value with defined start, end and interval'''
    
    list_ = list(range(int(start*100),int(end*100),int(step*100))) + [end*100]
    results = [float("{0:.2f}".format(x*0.01)) for x in list_]
    return(results)