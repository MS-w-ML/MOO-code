
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections.abc
from abc import ABC, abstractmethod
import time, datetime
from itertools import compress
from sklearn import metrics  
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff ()
if __name__=='__main__':
    import sys
    sys.path[0] = 'e:\\projects\\MOO'
from dataio.database import Database
from dataio.utils import save_csv, list_generator
from simulator.functions import get_function_class
from simulator.ml_learner import ML_Modelling
from simulator.utils import  construct_global_space
from dataio.utils import load_XY, append_csv
from utils.utils import recursive_mkdir, normalize


class Base_SyntheticDataset(ABC): 
    def __init__(self, opt):
        
        self.opt = opt
        self.fix_init_tr = opt.dataset.fix_init_tr if hasattr(opt.dataset, 'fix_init_tr') else True 
        self.isSynthetic = (opt.mode=='synthetic')
    
    def initialize(self):  
        '''Loop through every generated condition to find min and max.'''

        if not self.isSynthetic: # real dataset
            # match index of init tr set to the search space
            assert hasattr(self.opt.dataset, 'init_tr_pth') and self.opt.dataset.init_tr_pth is not None
            df = pd.read_csv(self.opt.dataset.init_tr_pth, index_col='index')
            df.index = list(range(df.shape[0]))

            ind_new_list = []
            replace_ind = -1
            for ind in df.index:
                ret_ind = self.get_index(df.loc[ind], replace_index=replace_ind)
                
                if ret_ind < 0:
                    ind_new_list += [replace_ind]
                    replace_ind -= 1
                else:
                    ind_new_list += [ret_ind]

            
            df.index = ind_new_list
            print('Training dataset loaded from %s, size=%d'% (self.opt.dataset.init_tr_pth, df.shape[0]))
            
            # replace
            df.to_csv(self.opt.dataset.init_tr_pth, index=True, index_label='index')
            
        else: # synthetic dataset
            # check if no normalization needed, exits
            ret_flag = True
            for y_lbl in self.y_lbl_list:
                cur_flag = (not self.norm_flag[y_lbl]) or (self.norm_flag[y_lbl] and not self.find_minmax_flag[y_lbl])
                ret_flag = ret_flag and cur_flag
            if ret_flag:
                return 
                
            res_dict = {}
            for y_lbl in self.y_lbl_list:
                res_dict[y_lbl+'_min'] = getattr(self, y_lbl+'_min')
                res_dict[y_lbl+'_max'] = getattr(self, y_lbl+'_max')

            # loop through dataset to find max
            start_time = time.time()
            counter = 0
            print('Start initializing dataset...',flush=True) #TODO: delete
            for item in self.batch_iterator(batch_size=int(1e7), normalize_y=False):
                print('%6d loop, time = %.2f mins'%(counter, (time.time()-start_time)/60), flush=True)
                for y_lbl in self.y_lbl_list:
                    if self.norm_flag[y_lbl] and self.find_minmax_flag[y_lbl]: # if need to normalize, and need to find minmax
                        y_max = np.max(item[y_lbl+'_real'])
                        y_min = np.min(item[y_lbl+'_real'])
                        if y_max > res_dict[y_lbl+'_max']:
                            res_dict[y_lbl+'_max'] = y_max

                        if y_min < res_dict[y_lbl+'_min']:
                            res_dict[y_lbl+'_min'] = y_min
                counter += 1
            # assign values attributes
            for y_lbl in self.y_lbl_list:
                setattr(self, y_lbl+'_min', res_dict[y_lbl+'_min'])
                setattr(self, y_lbl+'_max', res_dict[y_lbl+'_max'])   

        
        print('Initialized %s with size %s.'%(str(self.__class__), len(self)))

    def get_index(self, x, replace_index):
        '''Given an experiment condition, output its index in the search space.
        If not found exactly matched condition, return -1
        '''
        
        feature_list = self.fake_x_rng_dict[0].keys()
        start_index = 0
        for i in range(len(self.fake_x_rng_dict)):
            index = 0
            found_flag = True
            for feature in feature_list:
                _feat_val_list = pd.DataFrame(self.fake_x_rng_dict[i][feature]['val'])
                found_inds = _feat_val_list.loc[_feat_val_list[0]==x[feature]].index
                if len(found_inds) == 0: 
                    found_flag = False
                    break
                feature_ind = found_inds[0]
                index += feature_ind * self.fake_x_rng_dict[i][feature]['count_divider'] + feature_ind

            if found_flag:
                index += start_index
                break
            else:
                start_index += self.fake_x_rng_dict_lens[i]

        if not found_flag:
            index = -1
        return int(index)

    def get_x(self, index, return_pd=True):
        '''Return corresponding experiment condition'''
        assert int(index) == index
        index_origin = int(index)

        start_x_index = 0
        rng_end_index = 0 
        for rng_index in range(len(self.fake_x_rng_dict_lens)):
            rng_end_index += self.fake_x_rng_dict_lens[rng_index]
            if index < rng_end_index:
                break
            else:
                start_x_index += self.fake_x_rng_dict_lens[rng_index]


        cur_row = []
        x_index_rng = index - start_x_index
        feature_list = self.fake_x_rng_dict[rng_index].keys()
        for feature in feature_list:
            _feat_val_list = self.fake_x_rng_dict[rng_index][feature]['val']
            feature_ind = int(x_index_rng // self.fake_x_rng_dict[rng_index][feature]['count_divider'])
            cur_row += [_feat_val_list[feature_ind]]
            x_index_rng -= feature_ind * self.fake_x_rng_dict[rng_index][feature]['count_divider']

        if return_pd:
            x = pd.DataFrame([cur_row], columns=feature_list, index=[index_origin])
        else:
            x = cur_row
        return x
    
    @abstractmethod
    def get_xy(self, index, normalize_y):
        pass

    def __len__(self):
        return self.global_space_size 
    
    def __iter__(self):  
        '''Loop through all experiment conditions defined by _global_space_rng'''

        for i in range(len(self)):
            item = self.get_xy(i, normalize_y=True)
            yield   {'index':i , **item}
    
    @abstractmethod
    def batch_iterator(self, batch_size, index_to_skip=[], normalize_y=True):
        '''Iterate through conditions that is not within index_to_skip'''
        pass

    def get_feature_list(self):
        return list(self.fake_x_rng_dict[0].keys())

    def get_y_lbl_list(self):
        return self.y_lbl_list
        
    def get_init_training_set(self, init_tr_size):  
        '''
        If mode='synthetic', and 1) if 'SO' or 'MOO' w/o ncolor constraint randomly: select conditions to construct the initial training set;
            else 2) if 'MOO' with ncolor constraint: choose conditions based on ncolors constraint.
        If mode='real', get all init tr set and make sure #exp is the same as init_tr_size.
        '''
        if not self.isSynthetic: # real
            dpath = self.opt.dataset.init_tr_pth
            if dpath.endswith("csv"):
                df = pd.read_csv(dpath, index_col='index')
            #elif dpath.endwith("xlsx"):
            #    df = pd.read_excel(dpath, index_col='index', sheet_name=0)
            else:
                raise Exception('Invalid initial training file format, shall ends with .csv or .xlsx: '+str(dpath))
            
            init_tr_set_inds = list(df.index)
            feature_list = list(self.fake_x_rng_dict[0].keys())
            x_df = df[feature_list]

            y_dict = {}
            for y_lbl in self.y_lbl_list:
                y_dict[y_lbl+'_real'] = list(df[y_lbl+'_real'])
            y_df = pd.DataFrame.from_dict(y_dict)
            assert x_df.shape[0] == init_tr_size

        else: # synthetic
            tr_ncolor = self.opt.dataset.tr_ncolor if (self.opt.dataset.name == 'MOO' and hasattr(self.opt.dataset, 'tr_ncolor')) else None
            if self.fix_init_tr:
                init_tr_prefix = self.opt.postfix if hasattr(self.opt, 'postfix') else 0
                # if 'SO' or 'MOO' w/o ncolor constraint
                if tr_ncolor is None:
                    init_index_pth = './data/simulator/synDS%d_initTr_c%df%d.txt'%(init_tr_prefix, init_tr_size, len(self)) 
                else:
                    init_index_pth = './data/simulator/synDS%d_initTr_c%df%d_nc%dt%d.txt'%(init_tr_prefix, init_tr_size,len(self),tr_ncolor, int(self.opt.dataset.yield_target*100)) 

                if not os.path.exists(init_index_pth):
                    init_tr_set_inds = self.gen_init_tr_set(init_tr_size, tr_ncolor)
                    np.savetxt(init_index_pth, init_tr_set_inds)
                    print('Saving init tr set to %s...'%init_index_pth)
                else:
                    init_tr_set_inds = np.loadtxt(init_index_pth) 
                    print('Loading init tr set from %s...'%init_index_pth)
            else:
                init_tr_set_inds = self.gen_init_tr_set(init_tr_size, tr_ncolor)
            
            x_df = None
            y_dict = {}
            for ind in init_tr_set_inds:
                item = self.get_xy(ind)
                if x_df is None:
                    x_df = item['x']
                    for y_lbl in self.y_lbl_list:
                        y_dict[y_lbl+'_real'] = [item[y_lbl+'_real']]
                else:
                    x_df = x_df.append(item['x'])
                    for y_lbl in self.y_lbl_list:
                        y_dict[y_lbl+'_real'] += [item[y_lbl+'_real']]

            y_df = pd.DataFrame.from_dict(y_dict)

        return init_tr_set_inds, x_df, y_df

    def gen_init_tr_set(self, init_tr_size, tr_ncolor):
        '''Randomly select a set of conditions as init training set defined by the init_tr_size. 
        And in the scenario of 'MOO', ncolor_found constraint may apply.
        
        params:
            init_tr_size: <int> size of init tr set
            tr_ncolor:    <int | None> ncolor_found constraint in training set for 'MOO'
        '''

        if tr_ncolor is None:
            init_tr_set_inds = random.sample(range(len(self)), init_tr_size) 
        else:
            # randomly set the colors that shall exists 
            tot_ncolor = len(list(self.color_cls_boundary.keys()))
            selected_color_lbls = random.sample(range(tot_ncolor), tr_ncolor)
            yield_target = self.opt.dataset.yield_target
            yield_threshold = {} # fake threshold for the selected colors
            for selected_color in selected_color_lbls:
                _threshold = random.gauss(0.5,0.3)
                while(_threshold > 1 or _threshold < 0):
                    _threshold = random.gauss(0.5,0.3)
                yield_threshold[selected_color] = _threshold

            print('yield_threshold', str(yield_threshold))

            # loop through each randomly selected index
            checked_index_list = []
            init_tr_set_inds = []

            while(len(init_tr_set_inds) < init_tr_size):
                next_ind = random.sample(range(len(self)), 1)[0]
                if next_ind not in checked_index_list:
                    item = self.get_xy(next_ind)
                    color_lbl = self.convert2colorlbl(item['color_real'])[0]
                    if item['yield_real'] <= yield_target or (color_lbl in selected_color_lbls and item['yield_real'] <= yield_threshold[color_lbl]): # check if the condition has a color in defined set
                        init_tr_set_inds += [next_ind]
                    checked_index_list += [next_ind]

        return init_tr_set_inds



    @abstractmethod
    def get_func(self, simulator_opt):
        '''parse func name and return the initiated function class'''
        pass


    def parse_global_rng(self, ds_opt):
        global_space_rng_pth = ds_opt.global_space_rng_pth if hasattr(ds_opt, 'global_space_rng_pth') else None
        if global_space_rng_pth is not None:
            load_path = global_space_rng_pth
        else:
            load_path = './data/simulator/global_space/%s_global_space.json'%self.dataset_name
        
        
        restrict_num_feat = ds_opt.restrict_num_feat if hasattr(ds_opt, 'restrict_num_feat') else None
        construct_global_space(self.dataset_name, load_path, y_type=None)
        self.parse_global_rng_from_json(load_path, restrict_num_feat)

        
    def parse_global_rng_from_json(self, json_file_pth, restrict_num_feat=None):

        with open(json_file_pth, encoding='utf-8') as f:
            dict_file = json.load(f)
            if(isinstance(dict_file, dict)):
                dict_file = [dict_file]

        self.fake_x_rng_dict  = []
        self.global_space_size = 0

        for rng_dict in dict_file:
            res_dict = {}
            feature_val_count_dict = {}
            if restrict_num_feat is not None:
                assert restrict_num_feat <= len(rng_dict.keys())

            
            for key in rng_dict.keys():
                val_list = []
                list_temp = rng_dict[key]

                for i in range(len(list_temp)):

                    dict_temp = list_temp[i]
                    if dict_temp['type'] == 'list_range':
                        _min = dict_temp['val']['min']
                        _max = dict_temp['val']['max']
                        _interv = dict_temp['val']['interval']
                        val_list += list_generator(_min, _max, _interv)

                    elif dict_temp['type'] == 'list_value':
                        val_list += dict_temp['val']

                    else:
                        raise Exception('Unidentified type defined \'%s\'. It should be of \'list_range\' or \'list_value\'. '% dict_temp['type'])
                
                res_dict[key] = {}
                res_dict[key]['val'] = val_list
                feature_val_count_dict[key] = len(val_list)
                # end loop

            if restrict_num_feat is None:
                self.nfeature = len(res_dict.keys())
                _fake_x_rng_dict = res_dict
            else:
                self.nfeature = restrict_num_feat
                _fake_x_rng_dict = {}
                feature_val_count_dict = {k: v for k, v in sorted(feature_val_count_dict.items(), key=lambda item: item[1], reverse=True)}

                for key in list(feature_val_count_dict.keys())[:restrict_num_feat]:
                    _fake_x_rng_dict[key] = res_dict[key]


            # add size of global space
            _space_size = 1
            for key in _fake_x_rng_dict.keys():
                _space_size *= feature_val_count_dict[key]


            self.fake_x_rng_dict += [_fake_x_rng_dict]
            self.global_space_size += _space_size
            

        
        print('Global space is defined as %d'%self.global_space_size)
        print(self.fake_x_rng_dict)




    def get_index_table(self):
        '''Indexing each condition given global space rng'''

        
        # compute count for each feature space
        self.fake_x_rng_dict_lens = []
        for i in range(len(self.fake_x_rng_dict)):
            _size = 1
            feature_list = list(self.fake_x_rng_dict[0].keys())
            for feature in feature_list:
                val_count = len(self.fake_x_rng_dict[i][feature]['val'])
                _size *= val_count
                self.fake_x_rng_dict[i][feature]['val_count'] = val_count
            
            self.fake_x_rng_dict_lens += [_size] 

            count_divider = 1
            feature_list.reverse()
            for feature in feature_list:
                self.fake_x_rng_dict[i][feature]['count_divider']  = count_divider
                count_divider *= self.fake_x_rng_dict[i][feature]['val_count']

        return






class SO_SyntheticDataset(Base_SyntheticDataset): 
    def __init__(self, opt):
        super().__init__(opt)
        
        self.y_lbl_list = ['y']
        self.dataset_name = opt.dataset.name
        self.y_type = opt.dataset.y_type

        # get global sapce
        self.parse_global_rng(opt.dataset)
 
        if self.isSynthetic:
            self.y_max, self.y_min = float('-inf'), float('inf') 
            self.f_y = self.get_func(opt.dataset.simulator_so)  



        self.get_index_table()
        self.y_max_new, self.y_min_new = 1, 0

        
        print('Constructing a SO synthetic dataset...')


    def get_func(self, simulator_opt):
        '''Generate pseudo functions as one of below:
            <simulator.functions.POLYFIT> 
            <simulator.ml_learners.ML_Modelling>
        whose .__call__() is used to make predictions on input conditions
        '''

        type_nm = simulator_opt.type 
        func_nm = simulator_opt.name

        overwrite = simulator_opt.overwrite if hasattr(simulator_opt, 'overwrite') else False
        self.norm_flag = {}
        self.norm_flag['y'] = False     

        
        # get functions or models
        self.func_save_title = '%s_%s_%s'%(type_nm, func_nm, self.dataset_name)
        args_plus = {} # additional arguments to save for self.eval_fit()
        args_plus['type_nm'] = type_nm
        args_plus['func_nm'] = func_nm
        if type_nm == 'functions' and func_nm.upper() == 'POLYFIT':

            deg = simulator_opt.degree
            model = get_function_class(func_nm)(self.dataset_name, deg, overwrite, y_type=self.y_type)
            self.func_save_title += '_deg%d'%deg
            args_plus['deg'] = deg
            save_title = type_nm[:5]+func_nm.upper()


        elif type_nm == 'ml_learner':
            model = ML_Modelling(ds_name=self.dataset_name, y_type=self.y_type, ml_model_name=func_nm)
            save_title = type_nm[:5]

            
        else:
            raise Exception('Invalid stimulator name: %s'%type_nm)
        
        self.eval_fit(model, args_plus, save_title=save_title)
        return model


    def get_xy(self, index, normalize_y=True):
        '''Return corresponding experiment condition and label with given index.'''
        x = self.get_x(index)
        res_dict = {}
        res_dict['x'] = x

        # get y(s)
        y = self.f_y(x)[0]
        if normalize_y and self.norm_flag['y']:
            res_dict['y_real'] = normalize(y, old_max=self.y_max, old_min=self.y_min, new_max=self.y_max_new, new_min = self.y_min_new)
        else:
            res_dict['y_real'] = y   

        return res_dict
    
    def batch_iterator(self, batch_size, index_to_skip=[], normalize_y=True):
        '''Iterate through conditions that is not within index_to_skip'''

        count = 0
        count_sum = 0
        tot = len(self)-len(index_to_skip)
        feature_list = self.fake_x_rng_dict[0].keys()
        batch_size = batch_size if batch_size < tot else tot
        x_mat = np.zeros((batch_size,len(feature_list)))
        ind_mat = np.zeros((batch_size,))        
        
        for ind in range(len(self)):
            if ind not in index_to_skip:

                x = self.get_x(ind, return_pd=False)
                x_mat[count,:] = np.array(x)
                ind_mat[count] = ind
                count += 1
                
                if count != 0 and count % batch_size == 0: # create a temp dataFrame
 
                    x = pd.DataFrame(data=x_mat, columns = feature_list, index = ind_mat) 
                    if not self.isSynthetic:
                        yield   {'index':ind_mat, 'x':x}
                        del x, ind_mat, x_mat
                    else:
                        y = self.f_y(x)
                        if normalize_y and self.norm_flag['y']:
                            y = normalize(y, old_max=self.y_max, old_min=self.y_min, new_max=self.y_max_new, new_min = self.y_min_new)
                        yield   {'index':ind_mat, 'x':x, 'y_real':y}
                        del x, y , ind_mat
                    count_sum += count
                    batch_size = batch_size if batch_size < (tot -count_sum) else tot-count_sum
                    if batch_size == 0:
                        break
                    count = 0  
                    x_mat = np.zeros((batch_size,len(feature_list)))
                    ind_mat = np.zeros((batch_size,))



    def eval_fit(self, f, args_plus, save_title):
        '''Evaluate the function f's fitting on dataset '''

        eval_res_pth = os.path.join('./simulator/fit_results/%s/%s/%s_eval_results.csv'%(self.dataset_name, self.y_type, save_title))
        recursive_mkdir(os.path.dirname(eval_res_pth))

        X, y_true = load_XY(self.dataset_name, as_matrix=False, y_type=self.y_type, norm=True)
        y_true = y_true.to_numpy()
        y_pred = f(X)

        res_dict = {}
        res_dict.update(args_plus)
        res_dict['dataset'] = self.dataset_name
        res_dict['y_type'] = self.y_type
        res_dict['func_save_title'] = self.func_save_title
        res_dict['r2'] = metrics.r2_score(y_true, y_pred)   
        res_dict['mse'] = metrics.mean_squared_error(y_true, y_pred)
        [res_dict['pearson'], res_dict['p_val']] = pearsonr(y_true, y_pred)
        

        append_csv(eval_res_pth, res_dict)
        #with open(eval_res_pth, 'w') as fp:
        #    json.dump(res_dict, fp)

        self.plot_fit(f, X, y_true, self.func_save_title)
        return res_dict
    

    def plot_fit(self, f, X, y_real, title):
        # PCA: compress X into 1 principal component
        pca_x = PCA(n_components=1)
        x_pca = pca_x.fit_transform(X)

        x_plot = np.linspace(min(x_pca), max(x_pca), 300).reshape(-1, 1)
        x_inv = pca_x.inverse_transform(x_plot)
        x_inv_df = pd.DataFrame(x_inv, columns=X.columns)
        y_plot = f(x_inv_df)

        save_path = os.path.join('./simulator/fit_results/%s/%s/%s.png'%(self.dataset_name, self.y_type, self.func_save_title))
        plt.figure()
        plt.scatter(x_pca, y_real)
        plt.plot(x_plot,y_plot,color="black")
        plt.title(title)
        #plt.show()
        plt.savefig(save_path)
        plt.close()





class MOO_SyntheticDataset(Base_SyntheticDataset): 
    def __init__(self, opt):
        super().__init__(opt)

        self.color_cls_boundary = opt.color_cls_boundary

        self.y_lbl_list = ['yield', 'color']

        self.dataset_name = opt.dataset.name

        # get global sapce
        self.parse_global_rng(opt.dataset)

        if self.isSynthetic:
            self.yield_max, self.yield_min = float('-inf'), float('inf') 
            self.color_max, self.color_min = float('-inf'), float('inf') 

            self.f_yield, self.f_color = self.get_func(opt.dataset.simulator_moo)  # generated pseudo functions <simulator.functions.AbstractFunction> or learned ML relationship <simulator.ml_learners.ML_Modelling>
                                            # whose .__call__() is used to make predictions on input conditions
        #else:
        #    self.parse_global_rng_from_json(opt.dataset.global_space_rng_pth)   
        self.get_index_table()

        self.yield_max_new, self.yield_min_new = 1, 0
        self.parse_color_maxmin() #TODO

        
        print('Constructing a MOO synthetic dataset...')

    def get_total_ncolor(self):
        return len(self.color_cls_boundary.keys())

    def parse_color_maxmin(self):
        color_min = np.float('inf')
        color_max = - np.float('inf')

        color_strings = list(self.color_cls_boundary.keys())
        for i in range(len(color_strings)):
            color_nm = color_strings[i] 
            _min = self.color_cls_boundary[color_nm]['min']
            _max = self.color_cls_boundary[color_nm]['max']
            assert _min <_max
            if _min < color_min:
                color_min = _min
                color_min_lbl = i
            if _max > color_max:
                color_max = _max
                color_max_lbl = i

        self.color_max_new = color_max
        self.color_min_new = color_min
        self.color_max_lbl = color_max_lbl
        self.color_min_lbl = color_min_lbl
        print('self.color_max_lbl = %d'%self.color_max_lbl)
        print('self.color_min_lbl = %d'%self.color_min_lbl)


    def colorstr2numlbl(self, color_strs):
        ''''Convert color strings to color label indexes, 
        e.g. 'red' -> 0, ['red', 'blue'] -> [0,1]'''
        if isinstance(color_strs, str):
                    confined_color_list = [color_strs]
        elif not isinstance(color_strs, collections.abc.Sequence):
            raise Exception('Invalid color_strs=%s. \
                Should be a list of color strings, e.g. [\'red\',\'blue\']'%str(color_strs))

        full_color_strings = list(self.color_cls_boundary.keys())
        out_color_index_list = []
        for color_str in color_strs:
            out_color_index_list += [full_color_strings.index(color_str)]

        return out_color_index_list
        
            

    def convert2colorlbl(self, y_color):
        '''Convert y_color to color label index, as defined in self.opt.color_cls_boundary.'''
        
        start_time = time.time()
        if not isinstance(y_color, np.ndarray):
            y_color = np.atleast_1d(y_color)

        y_len = y_color.shape[0]
        assert len(y_color.shape) == 1
        color_lbls = np.zeros(y_len).astype(int) - 1 # construct an array of -1 
        color_lbl_list = list(self.color_cls_boundary.keys())
        ncolor = len(color_lbl_list)

        for i in range(ncolor):
            color_lbl = color_lbl_list[i]
            _left = self.color_cls_boundary[color_lbl]['min']
            _right = self.color_cls_boundary[color_lbl]['max']
            _is_in_range = np.logical_and(y_color >= _left , y_color < _right)
            _on_rightest_boundary = np.logical_and(i == (ncolor-1) , y_color == _right) 
            bool_arr = np.logical_or(_is_in_range, _on_rightest_boundary)
            color_lbls[bool_arr] = i

        # handle those values outside valid range, e.g. < color_min, or > color_max
        color_lbls[np.logical_and(color_lbls==-1, y_color<self.color_min_new)] = self.color_min_lbl
        color_lbls[np.logical_and(color_lbls==-1, y_color>self.color_max_new)] = self.color_max_lbl

        tot_time = (time.time() - start_time) / 60
        if self.opt.verbose:
            print('convert2colorlbl - time = %.2f'%tot_time)
        return color_lbls




    def get_func(self, simulator_opt):
        '''1) parse func name and return the initiated function class.
            2) set global search space
        '''

        yield_opt = simulator_opt.f_yield
        color_opt = simulator_opt.f_color

        yield_type = yield_opt.type 
        color_type = color_opt.type
        self.norm_flag = {}
            
        self.norm_flag['color'] = True
        if yield_type == 'ml_learner' and color_type == 'ml_learner':
            assert yield_opt.ds_name == color_opt.ds_name, 'MOO synthetic datasets needs two functions of type \'ml_learner\' to be constructed from the same dataset.'
            f_yield = ML_Modelling(ds_name=yield_opt.ds_name, ml_model_name=yield_opt.name)
            f_color = ML_Modelling(ds_name=color_opt.ds_name, ml_model_name=color_opt.name)
            _global_space_rng_pth = f_yield.global_space_rng_pth
            self.parse_global_rng_from_json(_global_space_rng_pth)

            self.norm_flag['yield'] = False
            

        elif yield_type == 'ml_learner' and color_type == 'functions':
            f_yield = ML_Modelling(ds_name=yield_opt.ds_name, ml_model_name=yield_opt.name)
            _global_space_rng_pth = f_yield.global_space_rng_pth
            self.parse_global_rng_from_json(_global_space_rng_pth)
            overwrite = color_opt.overwrite if hasattr(color_opt, 'overwrite') else False
            if color_opt.name.upper() != 'RANDOM':
                f_color = get_function_class(color_opt.name)(self.nfeature, name=color_opt.title, overwrite=overwrite)
            else:
                f_color = get_function_class(color_opt.name)(self.global_space_size, name=color_opt.title, overwrite=overwrite)

            self.norm_flag['yield'] = False

            self.find_minmax_flag = {} 
            self.find_minmax_flag['color'] = True if color_opt.name.upper() != 'RANDOM' else False
            if not self.find_minmax_flag['color']:
                self.color_max, self.color_min = 1, 0

        elif yield_type == 'functions' and color_type == 'ml_learner':
            f_color = ML_Modelling(ds_name=color_opt.ds_name, ml_model_name=color_opt.name)
            _global_space_rng_pth = f_color.global_space_rng_pth
            self.parse_global_rng_from_json(_global_space_rng_pth)
            overwrite = yield_opt.overwrite if hasattr(yield_opt, 'overwrite') else False
            if yield_opt.name.upper() != 'RANDOM':
                f_yield= get_function_class(yield_opt.name)(self.nfeature, name=yield_opt.title, overwrite=overwrite)
            else:
                f_yield = get_function_class(yield_opt.name)(self.global_space_size, name=yield_opt.title, overwrite=overwrite)


            self.norm_flag['yield'] = True
            self.find_minmax_flag = {} 
            self.find_minmax_flag['yield'] = True if yield_opt.name.upper() != 'RANDOM' else False
            if not self.find_minmax_flag['yield']:
                self.yield_max, self.yield_min = 1, 0
            self.find_minmax_flag['color'] = True
                
        elif yield_type == 'functions' and color_type == 'functions':
            # get global space range
            _global_space_rng_pth = yield_opt.global_space_rng_pth
            restrict_num_feat = yield_opt.restrict_num_feat if hasattr(yield_opt, 'restrict_num_feat') else None
            assert (not hasattr(color_opt, 'global_space_rng_pth')) or _global_space_rng_pth == color_opt.global_space_rng_pth
            self.parse_global_rng_from_json(_global_space_rng_pth, restrict_num_feat)
            c_overwrite = color_opt.overwrite if hasattr(color_opt, 'overwrite') else False
            y_overwrite = yield_opt.overwrite if hasattr(yield_opt, 'overwrite') else False
            if yield_opt.name.upper() != 'RANDOM':
                f_yield= get_function_class(yield_opt.name)(self.nfeature, name=yield_opt.title, overwrite=y_overwrite)
            else:
                f_yield = get_function_class(yield_opt.name)(self.global_space_size, name=yield_opt.title, overwrite=y_overwrite)

            if color_opt.name.upper() != 'RANDOM':
                f_color = get_function_class(color_opt.name)(self.nfeature, name=color_opt.title, overwrite=c_overwrite)
            else:
                f_color = get_function_class(color_opt.name)(self.global_space_size, name=color_opt.title, overwrite=c_overwrite)


            self.norm_flag['yield'] = True
            self.find_minmax_flag = {} 
            self.find_minmax_flag['yield'] = True if yield_opt.name.upper() != 'RANDOM' else False
            if not self.find_minmax_flag['yield']:
                self.yield_max, self.yield_min = 1, 0
            
            self.find_minmax_flag['color'] = True if color_opt.name.upper() != 'RANDOM' else False
            if not self.find_minmax_flag['color']:
                self.color_max, self.color_min = 1, 0
                

        else:
            raise Exception('Invalid MOO stimulator function types defined.')

        return f_yield, f_color

    def get_xy(self,index, normalize_y=True):
        '''Return corresponding experiment condition and label with given index.'''
        x = self.get_x(index)
        res_dict = {}
        res_dict['x'] = x

        res_dict['yield_real'] = self.f_yield(x)[0]
        res_dict['color_real'] = self.f_color(x)[0]
        # get y(s)
        if normalize_y:
            if self.norm_flag['yield']:
                res_dict['yield_real'] = normalize(res_dict['yield_real'], old_max=self.yield_max, old_min=self.yield_min, new_max=self.yield_max_new, new_min = self.yield_min_new)
            
            if self.norm_flag['color']:
                res_dict['color_real'] = normalize(res_dict['color_real'], old_max=self.color_max, old_min=self.color_min, new_max=self.color_max_new, new_min = self.color_min_new)

        return res_dict     


    def batch_iterator(self, batch_size, index_to_skip=[], normalize_y=True):
        '''Iterate through conditions that is not within index_to_skip'''

        count = 0
        count_sum = 0
        tot = len(self) - len(index_to_skip)
        batch_size = batch_size if batch_size < tot else tot
        init_time = time.time()
        feature_list = self.fake_x_rng_dict[0].keys()
        x_mat = np.zeros((batch_size,len(feature_list)))
        ind_mat = np.zeros((batch_size,))

        for ind in range(len(self)):
            if ind not in index_to_skip:
                x = self.get_x(ind, return_pd=False)
                x_mat[count,:] = np.array(x)
                ind_mat[count] = ind
                count += 1      

                if count != 0 and count % batch_size == 0: # create a temp dataFrame
                    x = pd.DataFrame(data=x_mat, columns = feature_list, index=ind_mat) 
                    if not self.isSynthetic:
                        yield   {'index':ind_mat, 'x':x}
                        del ind_mat, x_mat, x
                    else:
                        yield_real = self.f_yield(x)
                        color_real = self.f_color(x)
                        if normalize_y:
                            if self.norm_flag['yield']:
                                yield_real = normalize(yield_real, old_max=self.yield_max, old_min=self.yield_min, new_max=self.yield_max_new, new_min = self.yield_min_new)
                            
                            if self.norm_flag['color']:
                                color_real = normalize(color_real, old_max=self.color_max, old_min=self.color_min, new_max=self.color_max_new, new_min = self.color_min_new)

                        yield   {'index':ind_mat, 'x':x, 'yield_real': yield_real, 'color_real': color_real}
                        del ind_mat, x_mat, x, yield_real, color_real

                    count_sum += count
                    
                    batch_size = batch_size if batch_size < (tot-count_sum) else (tot-count_sum)
                    if batch_size == 0:
                        break
                    count = 0
                    
                    x_mat = np.zeros((batch_size,len(feature_list)))
                    ind_mat = np.zeros((batch_size,))


if __name__ == '__main__':
    from configs import ConfigManager
    cm = ConfigManager('./configs/config_pam.json')
    opts = cm.get_options()
    base_ds = SO_SyntheticDataset(opts)
    real_dict = base_ds.parse_global_rng_from_json("./data/simulator/fake/moo_real.json")