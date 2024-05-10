import os
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import pearsonr
from abc import ABC, abstractmethod
from sklearn import metrics  
from collections.abc import Iterable 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from dataio.utils import append_csv

import warnings
warnings.filterwarnings("ignore")
import shutup; shutup.please()


class BaseModel(ABC):
    '''BaseModel to handle operation like .fit() on <dataio.Database> objects'''

    def __init__(self, opt, ds):

        self.opt = opt
        self.ds = ds
        self.result_lbl_list = None # result stats names required for each model; used in self.save2file()
        self.util_name = None # the utility key of predicted results used to rank the unexplored conditions,
                              # Only used in SO models

        opt_model = opt.model
        self.init_tr_size = int(opt_model.init_training_size)
        self.batch_size = int(opt_model.batch_size)
        self.chk_pth = opt.experiment_dir
        self.data_pth = os.path.join(self.chk_pth, opt_model.name.lower()+'_data.csv')
        self.eval_res_pth = os.path.join(self.chk_pth, opt_model.name.lower()+'_evalResult.csv')
        self.hyperparams_sc_intvl = opt_model.hyperparams_sc_intvl if hasattr(opt_model, 'hyperparams_sc_intvl') else 1 
        self.isMultithreading = self.opt.use_multithreading if hasattr(self.opt, 'use_multithreading') else False
        self.cat2onehot_cols = opt_model.cat2onehot_cols if hasattr(opt_model, 'cat2onehot_cols') else None
        self.cat2onehot_encs = None

        # parse surrogate model list
        surrogate_model_list = opt_model.surrogate_model.split(' ') #['xgb', 'mlp']
        for ind in range(len(surrogate_model_list)):
            model_name = surrogate_model_list[ind].lower()
            assert model_name in ['xgb', 'mlp', 'svm', 'polyfit']
            surrogate_model_list[ind] = model_name 
        self.surrogate_model_list = surrogate_model_list


        
    def initialize(self):
        '''Perform everything after __init__'''

        # init result names for self.save2file()
        self.save_name_list = []
        for y_lbl in self.ds.get_y_lbl_list():
            for res_lbl in self.result_lbl_list:
                self.save_name_list += [y_lbl +'_'+res_lbl]
            if hasattr(self, 'additional_lbl_list'):
                self.save_name_list += getattr(self, 'additional_lbl_list')

        # prepare initial training data
        if not (self.opt.continue_train and os.path.exists(self.data_pth)): # check if the data file exists
            ##  get initial training data and copy to checkpoint
            inds, x, y_real = self.ds.get_init_training_set(self.init_tr_size)
            self.save2file(0, inds, x, y_real=y_real)
            self.parse_init_tr()

        df = pd.read_csv(self.data_pth, index_col='index')
        self.epoch = np.max(df['epoch'])
        self.parse_prev_data(df)
    
        print('Initialized %s modelling.'%str(self.__class__))


    def get_eval_results(self, epoch=None): #TODO: delete
        '''Return evaluation results for each epoch util the current epoch.'''

        eval_res_df = pd.read_csv(self.eval_res_pth)

        if epoch is not None:
            if isinstance(epoch, int) or isinstance(epoch, float):
                epoch = [int(epoch)]
            else:
                assert isinstance(epoch, Iterable)
                eval_res_df = eval_res_df.loc[eval_res_df['epoch'].isin(epoch)]

        return eval_res_df


    def get_current_epoch(self):
        return self.epoch


    def parse_init_tr(self):
        pass 


    def parse_prev_data(self, df):
        pass


    def save_ml_model(self, model, name):

        if self.save_hyperparams_flag:
            fn = os.path.join(self.opt.experiment_dir,"%s.json"%name)
            with open(fn,"w") as f:
                params = model.get_params()
                json.dump(params, f)
            print('Saved model params to %s ...'%fn)


    def reset_ml_model(self, name, mlmodel_type='xgb'):

        # check whether to perform hyperparameter searching
        if self.epoch ==0 or (self.epoch+1) % self.hyperparams_sc_intvl == 0:

            # need to search for best hyperparams
            reg_final = get_ml_cv(model_name=mlmodel_type, inner_nsplit=10, verbose=False, n_jobs=4, reduce_params=True, debug=self.opt.debug)
            self.save_hyperparams_flag = True
        else:
            self.save_hyperparams_flag = False
            # load hyperparams
            fn = "%s.json"%name
            with open("%s.json"%name, 'rb') as f:
                params = json.load(f)
                reg_final = xgb.XGBRegressor(**params)
            print('Loaded model params from %s ...'%fn)

        return reg_final
        

    def save2file(self, epoch, indexes, x,  **kwargs): 
        '''Save the explored conditions to file'''

        x.index = indexes # reset the index of dataframe to the indexes of explored conditions

        kwargs_key_list = []
        if len(kwargs) > 0:
            kwargs_key_list =  list(kwargs.keys())
        for res_key in self.save_name_list: 
                for k_key in kwargs_key_list:
                    data = kwargs[k_key]
                    if isinstance(data, pd.DataFrame): # handle dataframe input 
                        for key in data.columns:
                            if res_key == key:
                                x[key] = data[key].to_numpy()
                    elif k_key == res_key:
                        x[res_key] = data
                
                if not res_key in x.columns:# fill columns not given as nan to align the file
                    temp_arr = np.empty((x.shape[0],))
                    temp_arr[:] = np.nan
                    x[res_key] = temp_arr


        x['epoch'] = np.zeros((x.shape[0],)) + int(epoch)

        # write to file
        if not os.path.exists(self.data_pth): # write header
            write_header = True    
        else:
            write_header = False
        
        x.to_csv(self.data_pth, mode='a', index=True, index_label='index',header=write_header)
        

    def update_row(self, index, **data_dict):
        '''Update a row with given index in data csv file.'''
        
        key_list =  list(data_dict.keys())
        known_exps = pd.read_csv(self.data_pth, index_col='index')
        for key in key_list:
            assert key in known_exps.columns
            known_exps.loc[index][key] = data_dict[key]

        # replace
        known_exps.to_csv(self.data_pth, index=True, index_label='index')
        

    def cat2onehot(self, df):
        '''For given columns, convert from categorical to onehot encoding'''

        if self.cat2onehot_cols is None:
            return df

        # get all cats for each col
        if self.cat2onehot_encs is None:
            self.cat_dict = {}
            self.cat2onehot_encs = {}
            for rng in self.ds.fake_x_rng_dict:
                for col in self.cat2onehot_cols:
                    if col not in self.cat_dict.keys():
                        self.cat_dict[col] = set()
                    self.cat_dict[col].update(rng[col]['val'])

            for col in self.cat2onehot_cols: 
                cat_set = np.array(list(self.cat_dict[col]))[:, np.newaxis]
                enc = OneHotEncoder()
                enc.fit(cat_set)
                self.cat2onehot_encs[col] = enc

        for col in self.cat2onehot_cols:
            
            cat_set = np.array(list(self.cat_dict[col]))[:, np.newaxis]
            data_in = df[col][:, np.newaxis]
            data_onehot = self.cat2onehot_encs[col].transform(data_in).toarray()
            df_onehot = pd.DataFrame(data_onehot, columns=[ col+'%d'%i for i in cat_set])
            df = df.drop(col, axis = 1).reset_index(drop=True)
            df = df.join(df_onehot)

        return df

        

    def eval(self, epoch=None, save=True):  
        '''Evaluate the model in the current epoch based on the selected experiments.
        Compute utility'''
        known_exps = pd.read_csv(self.data_pth, index_col='index')
        res_dict = {}

        if epoch is None:
            res_dict['epoch'] = self.epoch 
        else:
            res_dict['epoch'] = epoch

        for y_lbl in self.ds.get_y_lbl_list():
            known_exps = known_exps.loc[known_exps[y_lbl+'_real'].notnull()] # filter out those rows with xx_real = null
            y_true = known_exps[y_lbl+'_real'].to_numpy()[self.init_tr_size:]
            y_pred = known_exps[y_lbl+'_pred'].to_numpy()[self.init_tr_size:]

            if self.epoch > 0: 
                
                # normalize ys to [0,1]
                _ylbl = None
                if y_lbl not in ['yield', 'color']:
                    if y_true.max() <= self.yield_max_new and y_true.min() >= self.yield_min_new:
                        _ylbl = 'yield'
                    elif y_true.max() <= self.color_max_new and y_true.min() >= self.color_min_new:
                        _ylbl = 'color'
                
                else:
                    _ylbl = y_lbl

                
                if _ylbl == 'yield':
                    real_max = self.yield_max_new
                    real_min = self.yield_min_new
                
                elif _ylbl == 'color':
                    real_max = self.color_max_new
                    real_min = self.color_min_new
    
                if _ylbl is not None:
                    print('Normalizing predicted %s to [0,1] for evaluation...'%_ylbl)
                    y_pred[y_pred < real_min] = real_min
                    y_pred[y_pred > real_max] = real_max
                    y_pred = (y_pred - real_min) / (real_max - real_min)

                # compute r2
                res_dict[y_lbl+'_r2'] = metrics.r2_score(y_true,y_pred)   
                res_dict[y_lbl+'_mse'] = metrics.mean_squared_error(y_true,y_pred)
                [res_dict[y_lbl+'_pearson'],res_dict[y_lbl+'_p_val']] = pearsonr(y_true,y_pred)
                res_dict[y_lbl+'_mape'] = metrics.mean_absolute_percentage_error(y_true, y_pred)
            else:
                res_dict[y_lbl+'_r2'] = np.nan
                res_dict[y_lbl+'_mse'] = np.nan
                res_dict[y_lbl+'_pearson'] = np.nan
                res_dict[y_lbl+'_p_val'] = np.nan
                res_dict[y_lbl+'_mape'] = np.nan


        # compute utility
        utility_dict = self.compute_utility(known_exps)
        res_dict.update(utility_dict)


        # save to log file
        if save:
            data = pd.DataFrame.from_dict(res_dict, orient='index').T
            write_header = True if not os.path.exists(self.eval_res_pth) else False
            data.to_csv(self.eval_res_pth, mode='a',index=True, index_label='index',header=write_header)

        return res_dict


    
    ##  Abstract Methods
    @abstractmethod
    def fit(self, X,Y):
        pass


    @abstractmethod
    def predict(self, X):
        pass


    @abstractmethod
    def gen_lbl_for_last_batch(self):
        '''Generate labels for the selected conditions in the last batch for synthetic dataset'''
        pass


    @abstractmethod
    def get_next_batch(self):
        '''Select the next batch of conditions to explore; save them with prediction.
            And update self.epoch.
        '''
        pass


    @abstractmethod
    def compute_utility(self, df):
        '''Compute the utility value'''
        pass



class BaseModel_SO(BaseModel):
    '''BaseModel to handle operation like .fit() on <dataio.Database> objects'''

    def __init__(self, opt, ds):
        super().__init__(opt, ds)


    def parse_init_tr(self):
        df = pd.read_csv(self.data_pth, index_col='index')
        assert 'y_real' in df.columns 
        

    def parse_prev_data(self, df):
        assert 'y_real' in df.columns 


    def compute_utility(self, df):
        '''Max value as utility.
        '''
        y_real = df['y_real']

        res_dict = {'utility': np.max(y_real)}
        return res_dict


    def validate_tr_data(self):
        '''Check tr data before calling model.fit().'''
        #known_exps = pd.read_csv(self.data_pth, index_col='index')
        #index_to_lbl = list(known_exps['y_real'][known_exps['y_real'].isnull()].index)
        
        #if self.opt.mode == 'real':
        #    assert len(index_to_lbl) == 0, 'For mode=\'real\', all conditions shall be labelled before calling model.fit()!'
        pass
        

    def gen_lbl_for_last_batch(self):
        '''Generate fake labels for the last batch, so that it can be training data for the next run.
        Used when mode is synthetic
        '''

        known_exps = pd.read_csv(self.data_pth, index_col='index')
        index_to_lbl = list(known_exps['y_real'][known_exps['y_real'].isnull()].index)
        
        if self.opt.mode == 'synthetic':
            for ind in index_to_lbl:
                item = self.ds.get_xy(ind)
                y_real = item['y_real']
                self.update_row(index=ind, y_real=y_real)
        else:
            assert len(index_to_lbl) > 0, 'For mode=\'real\', selected new conditions shall not be labelled '
            # insert real color labels given color_real
             

    def get_next_batch(self):
        '''Get the next batch of recommended conditions'''
        known_exps = pd.read_csv(self.data_pth, index_col='index')
        known_inds = known_exps.index[known_exps.index >=0] # filter out those indexes that are not in search space
        assert len(known_inds) == len(np.unique(known_inds))

        batch = {}
        batch_count = 0
        for item in self.ds.batch_iterator(batch_size=int(1e7), index_to_skip=known_inds, normalize_y=True):
            print('batch %d...'%batch_count)
            x = item['x']
            pred_res = self.predict(x)

            batch_size = self.batch_size if x.shape[0] > self.batch_size else x.shape[0]
            best_pos_arr_ind = np.argsort(-pred_res[self.util_name])[:batch_size]
            #best_pos_ind = best_pos_arr_ind[best_pos_arr_ind] #TODO: ??

            for i in best_pos_arr_ind:
                ind = int(item['index'][i])

                if not batch: # empty 
                    batch['ind'] = [ind]
                    for key in pred_res.keys():
                        batch[key] = [pred_res[key][i]]
                elif len(batch['ind']) < self.batch_size:
                    batch['ind'] += [ind]
                    for key in pred_res.keys():
                        batch[key] += [pred_res[key][i]]
                else: # full
                    pred_util = pred_res[self.util_name][i]
                    if (pred_util > np.array(batch[self.util_name])).any(): # replace if better
                        min_ind = np.argmin(np.array(batch[self.util_name])) 
                        batch['ind'][min_ind] = ind
                        for key in pred_res.keys():
                            batch[key][min_ind] = pred_res[key][i]

            batch_count += 1

        self.epoch += 1
        # save the best batch
        for i in range(len(batch['ind'])):
            ind = batch['ind'][i]
            x = self.ds.get_x(ind)
            res_dict = {}
            for key in batch.keys():
                if key != 'ind':
                    res_dict['y_'+key] = batch[key][i] 
            self.save2file(self.epoch, [ind], x,  **res_dict)


 


class BaseModel_MOO(BaseModel):
    '''BaseModel to handle MOO scenario'''

    def __init__(self, opt, ds):
        super().__init__(opt, ds)

        self.color_cls_boundary = opt.color_cls_boundary # include left, exclude right (except for the last range)
        
        # {'yellow':{'min':100, 'max':200}, 
        #                             'red':{'min':200, 'max':350},
        #                             'blue':{'min':350, 'max':500}}
        self.yield_target = opt.dataset.yield_target # the threshold yield for each color; if it is reached, the color is considered as found
        self.color_found_flag = np.array([False] * len(self.color_cls_boundary.keys()))
        self.additional_lbl_list = ['color_lbl_real','color_lbl_pred']

    def parse_init_tr(self):
        df = pd.read_csv(self.data_pth, index_col='index')
        assert all(key_name in df.columns for key_name in ['color_real', 'yield_real'])
        
        df['color_lbl_real'] = self.ds.convert2colorlbl(df['color_real'])# convert to color labels

        df.to_csv(self.data_pth, index=True, index_label='index')
        

    def parse_prev_data(self, df):
        assert all(key_name in df.columns for key_name in ['color_real', 'yield_real'])

        # set color_found_flag
        for color_idx in range(len(self.color_cls_boundary.keys())):
            out1 = df['color_lbl_real'] == color_idx  
            out2 = df['yield_real'] > self.yield_target
            if np.logical_and(out1, out2).any():
                self.color_found_flag[color_idx] = True


    def get_final_eval_result(self):
        '''Return a dictionary of evaluation metrics upon completion of all epochs'''
        
        eval_res_df = pd.read_csv(self.eval_res_pth)
        total_epoch = self.opt.model.nepoch

        # process res_df to demanded metrics
        # 1) loop thr colors, 2) util, 3) ncolor_found, 4) nexperiments to find all colors
        color_keys = list(self.color_cls_boundary.keys())
        tmp_dict = {} 
        column_name_list = []
        for color_key in color_keys:
            column_name_list += [color_key+'_max']
        column_name_list += ['utility', 'ncolor_found']

        for col_nm in column_name_list:
            tmp_dict['ep0_'+col_nm] = eval_res_df[eval_res_df['epoch']==0][col_nm].values[0]
            tmp_dict['ep%d_'%total_epoch+col_nm] = eval_res_df[eval_res_df['epoch']==total_epoch][col_nm].values[0]
            tmp_dict['delta_'+col_nm] = tmp_dict['ep%d_'%total_epoch+col_nm] - tmp_dict['ep0_'+col_nm]

        # 4) nexperiments to find all colors
        total_ncolor = len(color_keys)
        nepoch_find_allcolor = eval_res_df[eval_res_df['ncolor_found'] == total_ncolor]['epoch'].values.min()
        tmp_dict['nexp_find_allcolor'] = nepoch_find_allcolor * self.batch_size + self.init_tr_size

        return tmp_dict


    def compute_utility(self, df):
        '''Compute the utility value'''

        color_strings = list(self.color_cls_boundary.keys())
        color_max_dict = self.find_color_max(df)

        # compute utillity
        utility = 0
        for color_ind in range(len(color_strings)):
            color_key = color_strings[color_ind]
            color_max = color_max_dict[color_key+'_max']
            if self.color_found_flag[color_ind]:
                color_max += 10
            utility += color_max
        color_max_dict['utility'] = utility
        color_max_dict['ncolor_found'] = np.sum(self.color_found_flag)
        return color_max_dict     


    def validate_tr_data(self):
        '''Check tr data before calling model.fit().'''
        #known_exps = pd.read_csv(self.data_pth, index_col='index')
        #index_to_lbl = list(known_exps['yield_real'][known_exps['yield_real'].isnull()].index)
        
        #if self.opt.mode == 'real':
        #    assert len(index_to_lbl) == 0, 'For mode=\'real\', all conditions shall be labelled before calling model.fit()!'
        pass


    def gen_lbl_for_last_batch(self):
        
        known_exps = pd.read_csv(self.data_pth, index_col='index')
        index_to_lbl = list(known_exps['yield_real'][known_exps['yield_real'].isnull()].index)
        
        if self.opt.mode == 'synthetic':
            
            for ind in index_to_lbl:
                item = self.ds.get_xy(ind)
                yield_real = item['yield_real']                           
                color_real = item['color_real']
                color_lbl_real = self.ds.convert2colorlbl(color_real)

                if yield_real >= self.yield_target:
                    self.color_found_flag[color_lbl_real] = True 
                self.update_row(index=int(ind), yield_real=yield_real,color_real=color_real, color_lbl_real=color_lbl_real)
        else:
            assert len(index_to_lbl) > 0, 'For mode=\'real\', selected new conditions shall not be labelled '


    def find_color_max(self, df):
        '''Find max yield of each color label'''


        yield_true, color_lbl_true = df['yield_real'], df['color_lbl_real']
        # initialize
        color_strings = list(self.color_cls_boundary.keys())
        color_max_dict = {}
        for color_key in color_strings:
            color_max_dict[color_key+'_max'] = 0

        # find max yield for each color
        color_lbl_found = np.unique(color_lbl_true, return_counts=False)
        color_lbl_found =  [int(i) for i in color_lbl_found if not np.isnan(i)]
        for color_key in color_lbl_found:
            max_temp = np.max(yield_true[color_lbl_true==color_key])
            color_max_dict[color_strings[color_key]+'_max'] = max_temp
    
        return color_max_dict





    ########## linked list #############
    class Node():
        def __init__(self, dataval=None):
            self.dataval = dataval
            self.next = None

    class SLinkedList():
        def __init__(self):
            self.head = None
            self.length = 0

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            if self.length == 0:
                raise Exception('The linkedlist is empty! Please insert nodes first.')

            counter = 0 
            node_ptr = self.head
            while node_ptr is not None:
                if index == counter:
                    return node_ptr
                node_ptr = node_ptr.next 
                counter += 1
            raise Exception('The requested node index %d exceeds the length of linked list %d'%(index, counter))

        def append(self, node_new):
            '''Append a node at the end of list'''
            end_ind =  len(self) - 1
            if end_ind == -1:
                self.insert(node_new,None)
            else:
                node_prev = self[end_ind]
                self.insert(node_new, node_prev)


        def insert(self, node_new, node_prev):
            '''Insert node after node_prev'''

            if node_prev is None: # as head node
                node_new.next = self.head
                self.head = node_new
            else:
                node_new.next = node_prev.next
                node_prev.next = node_new
            self.length += 1

        def replace(self, node_new, index):
            
            node_old = self[index]
            node_new.next = node_old.next 
            if index == 0:
                self.head = node_new
            else:
                node_prev = self[index-1]
                node_prev.next = node_new
            del node_old





####################################################################################################
# Helper classes and functions
####################################################################################################
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold



from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


from sklearn.utils import resample 
from sklearn.model_selection import cross_val_score

def get_surrogate_model_cv(name, ncv_split=10, verbose=False):
    '''Output either Polyfit or ML models based on name'''
    name = name.upper()
    if name == 'POLYFIT':
        model = get_polyfit_cv(ncv_split, verbose)
        
    elif name in ['XGB', 'SVM', 'MLP']: 
        model = get_ml_cv(name, inner_nsplit=ncv_split,verbose=verbose)
    return model



def get_polyfit_cv(inner_nsplit,verbose):
    
    reg = Pipeline([         
            ('polyfeat', PolynomialFeatures()),    
                ('sc', StandardScaler()), 
                ('linreg', LinearRegression() )
            ])

    inner_cv = KFold(n_splits=inner_nsplit, shuffle=False) 

    tuned_parameters =dict(polyfeat__degree=[2,5,8,11,14])
    cv = GridSearchCV(reg, tuned_parameters, cv=inner_cv, scoring='r2',verbose=verbose,n_jobs=4)
    return cv 
        

def get_ml_cv(model_name='mlp', inner_nsplit=10, verbose=False, n_jobs=4, reduce_params=True, debug=False):
        assert model_name.lower() in ['mlp', 'xgb', 'svm']        
        model_name = model_name.lower()


        # construct predictor
        inner_cv = KFold(n_splits=inner_nsplit, shuffle=False) 
        ml_reg = get_model(model_name, 'reg') 
        tuned_parameters = get_hyperparams_grid(model_name, 'reg', reduce_params, debug)
        ml_cv = GridSearchCV(ml_reg, tuned_parameters, cv=inner_cv, scoring='r2',verbose=verbose,n_jobs=n_jobs)
        return ml_cv
    
#TODO reg_featureimp = others.extract_feature_importance(self.reg_model, self.model_name, X, title+'_reg', sort=True, to_dir=to_dir)
    


def get_model(model_name, model_type, init_params=None):
    '''Return model base on name and type.
    
    params:
        model_name: str. ['mlp' |'xgb' | 'svm']
        model_type: str. ['clf'|'reg']
    '''
    model_name = model_name.lower()
    model_type = model_type.lower()
    assert model_name in ['mlp', 'xgb', 'svm']
    assert model_type in ['clf', 'reg']

    model_init_params = {}
    if init_params is not None:
        model_init_params.update(init_params)


    if model_name == 'mlp':
        if 'solver' in model_init_params.keys() and model_init_params['solver'] == 'lbfgs':
            model_init_params['max_iter'] = 5000 # increase number of iters for lbfgs
        else:
            model_init_params['max_iter'] = 500
            
        model_init_params.update({'early_stopping':True})
        ml_model = MLPRegressor
    elif model_name == 'svm':
        ml_model = SVR 
    elif model_name == 'xgb':
        model_init_params.update({'objective': "reg:squarederror",
                                  'min_child_weight':1,'tree_method':'exact', #'tree_method':"gpu_hist"
                                "n_jobs":4,"random_state":3,'seed':3, 'gpu_id':0})
        ml_model = XGBRegressor

    if model_name in ['mlp', 'svm']:
        return Pipeline([            
                    ('sc', StandardScaler()), 
                    (model_type, ml_model(**model_init_params) )
                    ])
    else:
        return ml_model(**model_init_params)
    


def get_hyperparams_grid(model_name, model_type,  reduce=True, debug=False):

    if model_name == 'mlp':
        params = dict(hidden_layer_sizes=[[5],[10],[20],[5,5],[10,10],[20,20],[5,5,5],[10,10,10],[20,20,20]],
                        max_iter=[5000],
                        alpha=[0, 1e-2, 1e-1,1], #L2 penalty (regularization term) parameter. TODO 
                        early_stopping=[True],
                        solver= [ 'adam', 'lbfgs'], #'lbfgs',
                        #activation=['tanh', 'relu'],
                        batch_size=[16, 32, 64, 'auto'],
                        learning_rate=['constant', 'invscaling', 'adaptive'],
                        #momentum=[0.7, 0.9], 
                        learning_rate_init=[1e-3, 1e-4]   #1e-3, 1e-4                   
                        )
    elif model_name =='svm':
        params = dict(kernel=['rbf',  'linear', 'sigmoid'],#'poly',
                                #Wdegree=[3,7,11],
                            C=[1e-2,1e-1,1,1e1,1e2],
                            epsilon=[0.1, 0.2],
                            gamma=[1e-2,1e-1,'auto', 'scale',1, 1e1,1e2]
                        ) 
    elif model_name == 'xgb':
        if debug:
            params = dict(learning_rate=[1e-2],
                            n_estimators=[500], #100,,300,400,500
                            colsample_bylevel = [0.7],
                        gamma=[0], #0,0.1,0.2,0.3,0.4
                        max_depth =[11], # [3,7,11]]
                        reg_lambda = [1], #[0.1,1,10]
                        # reg_alpha = [1],
                        subsample=[1])
        elif reduce: 
            params = dict(learning_rate=[1e-2, 1e-3],
                                n_estimators=[100,300,500,700,900], 
                                colsample_bylevel = [0.5,0.7,0.9], 
                                max_depth =[3,5,7,9,11,13], # 
                                gamma=[0,0.2], #
                                reg_lambda = [0.1,1,10], #
                                #reg_alpha = [1],
                                #subsample=[0.6,0.8,1]
                                )#
  
        else:
            params = dict(
                        learning_rate=[1e-2, 1e-3],
                        n_estimators=[100, 300, 500, 700],
                        colsample_bytree=[0.8, 1],
                        colsample_bylevel = [0.4, 0.6, 0.8, 1],
                        gamma=[0,0.1,0.2,0.3,0.4],
                        max_depth =[4,7,10],
                        reg_lambda = [0.1, 1,10],
                        subsample=[0.4, 0.6, 0.8, 1]
                        )

    if model_name in ['mlp', 'svm']:
        new_params = {}
        for key, val in params.items():
            new_params[model_type+'__'+key] = val
        params = new_params
    return params

def parse_hyperparams_cv(model_name, params):
    if model_name in ['mlp', 'svm']:
        new_params = {}
        for key, val in params.items():
            if key[:5] == 'reg__':
                new_key = key.split('__')[1]
                new_params[new_key] = val
        params = new_params

        if 'hidden_layer_sizes' in params.keys():
            hidden_layer_sizes = params['hidden_layer_sizes'].split(',')
            params['hidden_layer_sizes'] = [int(a) for a in hidden_layer_sizes]


    return params


def parse_hyperparams_bo(model_name, params):
    '''Convert params to model's valid range'''

    int_round_list = []
    if model_name=='mlp':
        # solver
        sovler_dict = {0: 'lbfgs', 1: 'adam'}
        ind = int(round(params['solver']))
        params['solver'] = sovler_dict[ind]
        # learning rate
        lr_dict = {1:'constant', 2:'invscaling', 3:'adaptive'}
        for i, val in lr_dict.items():
            if params['learning_rate'] <=i:
                params['learning_rate'] = val
                break
        # hidden_layer_sizes
        params['hidden_layer_sizes']  = [int(round(params['hidden_layer_size']))] * int(round(params['hidden_layer_num']))
        del params['hidden_layer_size'], params['hidden_layer_num']

        int_round_list = ['batch_size']

    elif model_name == 'xgb':
        int_round_list = ['n_estimators', 'max_bin', 'max_depth']

    elif model_name == 'svm':
        kernel_dict = {1:'rbf',  2:'linear', 3:'sigmoid'}
        for i, val in kernel_dict:
            if params['kernel'] <=i:
                params['kernel'] = val
                break
    elif model_name == 'polyfit':
        int_round_list = ['degree']


    # convert float to integer
    for item in int_round_list:
        if item in params.keys():
            params[item] = int(round(params[item]))

    return params

def get_hyperparams_bo(model_name):
    if model_name == 'mlp':    
        res_dict =  dict(hidden_layer_size=(2,128),
                        hidden_layer_num=(1,128),

                        alpha=(1e-4, 1), #L2 penalty (regularization term) parameter.
                        solver= (0,1),
                        batch_size=(4,64),
                        learning_rate=(0, 3),
                        learning_rate_init=(0.0001,0.001)                     
                        )
    elif model_name == 'xgb':
        res_dict = dict(
                        learning_rate=(0.001, 0.1),
                        n_estimators=(10, 700),
                        colsample_bytree=(0.5, 1),
                        colsample_bylevel = (0.5, 1),
                        gamma=(0.01,0.1),
                        max_depth =(1,60),
                        reg_alpha = (0.01, 10),
                        reg_lambda = (0.01, 10),
                        subsample=(0.5, 1),
                        min_child_weight=(0.0001, 30)
                        )
    elif model_name == 'svm':
        res_dict = dict(
                            kernel=(0,3),
                            C=(1e-2,1e2),
                            epsilon=(0.1, 0.2),
                            gamma=(1e-2,1e2)
                        ) 
    elif model_name == 'polyfit':
        res_dict = dict(degree=(1,30)
                        ) 
    return res_dict
        

def get_bo_bestparams(X,Y, surrogate_model_list, ncvsplit, verbose, cutoff,  total_runs=1000):

    params_dict = {}
    for model_name in surrogate_model_list:
        param_table = get_bo_bestparams_model(X,Y, model_name, ncvsplit, verbose, total_runs)
        params_dict[model_name] = param_table.iloc[:cutoff]
    
    return params_dict



def get_bo_bestparams_model(X,Y, surrogate_model_name, ncvsplit, verbose, total_runs=1000):
    from bayes_opt import BayesianOptimization
    
    def train_model(**params):

        params = parse_hyperparams_bo(surrogate_model_name, params)
        model = get_model(surrogate_model_name, 'reg', params)
        cv_scores = cross_val_score(model, X, Y, scoring='r2', cv=ncvsplit, n_jobs=4)
            
        return cv_scores.mean()
    
    bounds = get_hyperparams_bo(surrogate_model_name)
    optimizer = BayesianOptimization(
        f=train_model,
        pbounds=bounds,
        random_state=3,
        verbose=verbose
    )
    print('[%s]Bayesian optimization for params, #runs = %d'%(surrogate_model_name, total_runs))
    optimizer.maximize(init_points = int(total_runs/2), n_iter= int(total_runs - int(total_runs/2))) #here you set how many iterations you want.
    table = pd.DataFrame(columns=bounds.keys())
    for res in optimizer.res:
        res_dict = {'target':[res['target']]}
        res_dict.update(res['params'])
        table=table.append(pd.DataFrame(res_dict),
                                        ignore_index=True)
    table=table.sort_values(by = ['target'], ascending=False).reset_index(drop=True)#sort the list start from the best results
    return table



def get_cv_bestparams(X, Y, surrogate_model_list, ncvsplit, cutoff, verbose):
    '''find best hyperparams, cv for surrogate model'''
    params_dict = {}
    for model_name in surrogate_model_list:
        ml_cv = get_surrogate_model_cv(model_name, ncvsplit, verbose)
        ml_cv.fit(X, Y)
        print(ml_cv.best_params_)
        print(ml_cv.best_score_)
        ml_cv_results = ml_cv.cv_results_
        mean_test_score_xgb = ml_cv_results['mean_test_score']
         
        selected_inds = np.argsort(-mean_test_score_xgb)[:cutoff]
        params_list = [ml_cv_results['params'][i] for i in selected_inds]
        best_scores = mean_test_score_xgb[selected_inds] 

        res_table = pd.DataFrame(columns=['target'] + list(params_list[0].keys()))
        for ind in range(len(selected_inds)):
            res_dict = {'target':[best_scores[ind]]}
            res_dict.update(params_list[ind])
            if 'reg__hidden_layer_sizes' in res_dict.keys():
                hidden_layer_sizes = [str(a) for a in res_dict['reg__hidden_layer_sizes']]
                res_dict['reg__hidden_layer_sizes'] = ','.join(hidden_layer_sizes)
            res_table=res_table.append(pd.DataFrame(res_dict),
                                        ignore_index=True)
        res_table=res_table.sort_values(by = ['target'], ascending=False).reset_index(drop=True)#sort the list start from the best results
        params_dict[model_name] = res_table
    return params_dict


def get_surrogate_models(X, Y, surrogate_model_list, params_dict, n_bootstrap, optim_method):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            Y = Y.to_numpy()
        # parse run time for each surrogate model
        model_nbootstrap_dict = {}
        base_count = np.ceil(n_bootstrap / len(surrogate_model_list))
        count_modulus = n_bootstrap % len(surrogate_model_list)
        for surrogate_model in surrogate_model_list:
            if  count_modulus > 0:
                model_nbootstrap_dict[surrogate_model] = base_count + 1
                count_modulus -= 1
            else:
                model_nbootstrap_dict[surrogate_model] = base_count

                
        # start getting surrogate models...
        surrogate_models = []
        tr_scores = 0
       
        # create n decision models
        for model_name in model_nbootstrap_dict.keys():
            for i in range(int(model_nbootstrap_dict[model_name])):
                train_ind = resample(list( range(0, X.shape[0])), n_samples=X.shape[0], replace=True, random_state=i) # generate bootstrap population indexes
                _train_x, _train_y = X[train_ind], Y[train_ind]
                
                params_table = params_dict[model_name]
                param_ind = i % params_table.shape[0]
                tr_scores += params_table.at[param_ind, 'target']
                params = params_table.iloc[param_ind].to_dict()
                del params['target']
                if optim_method == 'grid':
                    model_params = parse_hyperparams_cv(model_name, params)
                elif optim_method == 'bayesian':
                    model_params = parse_hyperparams_bo(model_name, params)
                new_model = get_model(model_name, 'reg', init_params=model_params)#clone(self.surrogate_model) # clone a unfitted version of the model
                
                new_model.fit(_train_x, _train_y)
                print('bootstrap %d, %s...'%(i, model_name))
                print(model_params)
                surrogate_models += [new_model]

        return surrogate_models, tr_scores / n_bootstrap