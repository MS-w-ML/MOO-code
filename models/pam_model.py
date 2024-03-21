import shap
import time
import random
import numpy as np
import pandas as pd

from .base_model import BaseModel_SO, BaseModel_MOO


class PAM_Model_SO(BaseModel_SO):
    '''Progressive adaptive modelling'''

    def __init__(self, opt, ds):
        super().__init__(opt,ds)
        
        self.result_lbl_list = ['real','pred'] # result stats names required for each model; used in self.save2file()
        self.util_name = 'pred' # the utility key of predicted results used to rank the unexplored conditions
    

    def fit(self):
        '''Reset ml model and train from scratch with given data.'''
        # 1. reset ml models
        self.ml_model = self.reset_ml_model('ml_model')

        # 2. get training data
        feature_list = self.ds.get_feature_list()
        known_exps = pd.read_csv(self.data_pth, index_col='index')

        # 3. model fitting 
        X,Y = known_exps[feature_list], known_exps['y_real']
        self.ml_model.fit(X,Y)

        # 4. save model
        if self.ml_model.__class__.__name__ == 'GridSearchCV': # get the ml with best hyperparams
            self.ml_model = self.ml_model.best_estimator_
        self.save_ml_model(self.ml_model, 'ml_model')


    def predict(self, X):
        return {'pred':self.ml_model.predict(X)}
    


class PAM_Model_MOO(BaseModel_MOO):
    '''Progressive adaptive modelling'''

    def __init__(self, opt, ds):
        super().__init__(opt,ds)
        
        self.result_lbl_list = ['real','pred'] # result stats names required for each model; used in self.save2file()
        self.util_name = 'pred' # the utility key of predicted results used to rank the unexplored conditions
        
    
    def fit(self):
        '''Reset ml model and train from scratch with given data.'''

        # 1. reset ml models
        self.yield_model = self.reset_ml_model('yield_model')
        self.color_model = self.reset_ml_model('color_model')

        # 2. get training data
        feature_list = self.ds.get_feature_list()
        known_exps = pd.read_csv(self.data_pth, index_col='index')

        # 3. model fitting for yield
        known_exps_yield = known_exps.loc[known_exps['yield_real'].notnull()] # filter out those has no yield

        self.yield_model.fit(known_exps_yield[feature_list], known_exps_yield['yield_real'])
        if self.yield_model.__class__.__name__ == 'GridSearchCV': # get the ml with best hyperparams
            self.yield_model = self.yield_model.best_estimator_ 

        # 4. model fitting for color
        known_exps_color = known_exps.loc[known_exps['color_real'].notnull()] # filter out those has no color info
        yield_real, color_real = known_exps_color['yield_real'], known_exps_color['color_real']
        # check color found flags
        color_lbls = self.ds.convert2colorlbl(color_real)
        for color_lbl in range(len(self.color_found_flag)):
            if (yield_real[color_lbls==color_lbl] >= self.yield_target).any():
                self.color_found_flag[color_lbl]=True
                
        self.color_model.fit(known_exps_color[feature_list], known_exps_color['color_real'])
        if self.color_model.__class__.__name__ == 'GridSearchCV': # get the ml with best hyperparams
            self.color_model = self.color_model.best_estimator_

        # 5. save model
        self.save_ml_model(self.yield_model, 'yield_model')
        self.save_ml_model(self.color_model, 'color_model')


    def predict(self, X):
        return {'yield_pred':self.yield_model.predict(X), 
                'color_pred':self.color_model.predict(X)}


    def get_next_batch(self): 
        '''Get the next batch of recommended conditions'''
        known_exps = pd.read_csv(self.data_pth, index_col='index') 
        known_inds = known_exps.index


        self.color_max_dict = self.find_color_max(known_exps)
        color_strings = list(self.color_cls_boundary.keys())

        next_batch = BaseModel_MOO.SLinkedList()
        counter = 0
        yield_util_max = int(self.ds.yield_max_new) + 1 

        ## Iterate the remaining experiment conditions in batch:
        for item in self.ds.batch_iterator(batch_size=int(1e7 if not self.opt.debug else 100), index_to_skip=known_inds, normalize_y=True):
            pred_all = self.predict(item['x'])
            yield_util = pred_all['yield_'+self.util_name]
            
            batch_size = self.batch_size if item['x'].shape[0] > self.batch_size else item['x'].shape[0]
            
            # Compute yield utility :
            # 1) compute yield utility of an unseen condition as [the predicted yield - the max yield of seen conditions for this color]
            # 2) If a color is not found yet (no seen condition of this color), raise its priority through add seen max yield utility to its yield utility
            if self.color_found_flag.any(): # perform if any colors is found 
                color_lbls = self.ds.convert2colorlbl(pred_all['color_pred'])# convert to color labels
                for i in range(len(self.color_found_flag)):
                    color_key = color_strings[i]
                    color_found_flag = self.color_found_flag[i]
                    yield_util[color_lbls == i] -= self.color_max_dict[color_key+'_max'] # minus prev found max of the color
                    if not color_found_flag: # not found color shall has higher priority
                        yield_util[color_lbls == i] += yield_util_max

            # handle confined output colors, 
            # e.g if only conditions that gives 'red' color will be considered, when confined_color_list = ['red']
            if hasattr(self.opt, 'confined_output_colors') and self.opt.confined_output_colors is not None:
                valid_cond_count = 0
                confined_color_lbls = self.ds.colorstr2numlbl(self.opt.confined_output_colors)
                for this_color_lbl in range(len(color_strings)):
                    if not (this_color_lbl in confined_color_lbls):
                        yield_util[color_lbls == this_color_lbl] = float('-inf')
                    elif valid_cond_count < batch_size:
                        valid_cond_count += np.sum(color_lbls == this_color_lbl)

                if valid_cond_count == 0: # no valid condiiton in this batch, skip
                    continue
                elif valid_cond_count < batch_size: # adjust batch size if the number of valid conditions less than batch size
                    batch_size = valid_cond_count

            # find the best conditions in the current batch
            best_pos_arr_ind = np.argsort(-yield_util)[:batch_size] # sort all items based on yield [:batch_size]

            # Update nodes holding the current best conditions 
            selected_count = 0
            for i in best_pos_arr_ind:

                cond_ind = item['index'][i]
                x_i = item['x'].loc[cond_ind].to_frame().T
                data_dict = {'index': cond_ind,
                            'x':x_i}
                pred_res = self.predict(x_i)
                pred_res['color_lbl_pred'] = self.ds.convert2colorlbl(pred_res['color_pred'])
                for key in pred_res.keys():
                    pred_res[key] = pred_res[key][0] 
                data_val = {**data_dict, **pred_res}

                if  len(next_batch) == 0:
                    new_node = BaseModel_MOO.Node(dataval=data_val)
                    next_batch.append(new_node)
                else: # linkedlist not empty, loop through conditions to compare
                    node_prev = None
                    node_ptr = next_batch.head
                    counter = 0
                    while node_ptr is not None:
                        this_is_better = self.compare_conditions(node_ptr.dataval, data_val)
                        if not this_is_better: # if this new condition is not better than the current node in node list, exit
                            break
                        node_prev = node_ptr
                        node_ptr = node_ptr.next 
                        counter += 1

                    new_node = BaseModel_MOO.Node(dataval=data_val)
                    if len(next_batch) < self.batch_size: # insert
                        next_batch.insert(node_new=new_node, node_prev=node_prev)
                    elif node_prev is not None:
                        next_batch.replace(new_node, index=counter-1)

        self.epoch += 1
        # save the best batch
        node_ptr = next_batch.head
        while node_ptr is not None:
            val_dict = node_ptr.dataval
            index, x = val_dict['index'], val_dict['x']
            del val_dict['index'], val_dict['x']

            self.save2file(self.epoch, [index], x, **val_dict)
            node_ptr = node_ptr.next 


    def compare_conditions(self, cond1, cond2):
        '''If cond2 is better than cond1, return True; else False
            color not found > color found
            than higher yield_EI  > lower yield_EI
        '''
        # handle conditions that fails
        is_c1_fail = self.is_growth_fail(cond1)
        is_c2_fail = self.is_growth_fail(cond2)

        if is_c1_fail and is_c2_fail:
            return False
        elif is_c1_fail and not is_c2_fail:
            return True
        elif not is_c1_fail and is_c2_fail:
            return False

        # compare color
        color_lbl_list = list(self.color_cls_boundary.keys())
        cond2_color_lbl = self.ds.convert2colorlbl(cond2['color_pred'])[0]
        cond1_color_lbl = self.ds.convert2colorlbl(cond1['color_pred'])[0]
        if self.color_found_flag[cond2_color_lbl] == False and  self.color_found_flag[cond1_color_lbl] == True:
            return True
        if self.color_found_flag[cond2_color_lbl] == True and  self.color_found_flag[cond1_color_lbl] == False:
            return False

        # compare yield
        cond2_yield_pred = cond2['yield_'+self.util_name] - self.color_max_dict[color_lbl_list[cond2_color_lbl]+'_max']
        cond1_yield_pred = cond1['yield_'+self.util_name] - self.color_max_dict[color_lbl_list[cond1_color_lbl]+'_max']
        if cond2_yield_pred > cond1_yield_pred:
            return True
        else:
            return False


    def is_growth_fail(self, cond):

        fail_flag = np.isnan(cond['yield_'+self.util_name]) or (cond['yield_'+self.util_name] <= 0)

        return fail_flag

