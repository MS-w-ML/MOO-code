import os
import argparse
import warnings
import pandas as pd

import sys
sys.path.insert(0, os.getcwd())
from  configs import ConfigManager
from simulator import get_ds
from models import get_model
from utils.utils import get_cur_datetime, convert_s2hr

import time


def run(args):  
    start_time = time.time() 

    # Parse input arguments
    configMgr = ConfigManager(args.config, verbose=args.debug, mode=args.mode)
    json_opts = configMgr.get_options()
    print('configuration time: %.3f mins'%((time.time()-start_time)/60), flush=True)  
    
    # init synthetic / real dataset
    ds_cls = get_ds(json_opts.dataset.type)
    ds = ds_cls(json_opts)
    print('get ds_cls time: %.3f mins'%((time.time()-start_time)/60), flush=True)  
    ds.initialize()
    print('initialize [%s] time: %.3f mins'%(ds.__class__.__name__, (time.time() - start_time)/60), flush=True)  

    
    # init model
    model_cls = get_model(json_opts)
    model = model_cls(json_opts, ds)
    model.initialize()

    # start searching....
    start_epoch = int(model.get_current_epoch()+1)
    total_epoch = json_opts.model.nepoch
    total_ncolors = ds.get_total_ncolor()
    start_time = time.time()
    for epoch in range(start_epoch, total_epoch + 1): # epoch 0 indicates init training set; thus real model works from epoch 1 to nepoch specified
        
        if epoch == 1 or((epoch + 1) % json_opts.model.eval_freq == 0): # start, end or every interval
            print('epoch %d / %d |  '%(epoch-1, total_epoch))
            res_dict = model.eval(epoch=epoch-1)
            msg = ''
            for key in res_dict:
                if key != 'epoch':
                    msg += key+' : %.3f,  '%res_dict[key]
            
            print('epoch time: %s  >> %s '%(convert_s2hr(time.time() - start_time), msg), flush=True) 
        
        start_time = time.time()
        model.validate_tr_data()
        model.fit()
        model.get_next_batch() # save next batch of recommend conds to file
        model.gen_lbl_for_last_batch() # generate synthetic labels for the selected batch if mode ='synthetic'
        
        if json_opts.mode == 'real':
            break
    
    # final evaluation
    #print('start_epoch=',start_epoch)
    end_flag = False
    if start_epoch == total_epoch+1 or (args.mode == 'synthetic' and  epoch == total_epoch) :
        end_flag = True
        res_dict = model.eval(epoch=total_epoch)
        msg = ''
        for key in res_dict:
            if key != 'epoch':
                msg += key+' : %.3f,  '%res_dict[key]
        print('epoch %d / %d |  '%(total_epoch, total_epoch) + msg)
        print('epoch time: %s'%convert_s2hr(time.time() - start_time), flush=True) 

    # summarize performance and save to specified file
    if args.save_pth is not None and end_flag:
        # write to file
        print('saving result to file: %s'%args.save_pth,flush=True)
        # save to log file
        eval_res_df = model.get_eval_results()
        tmp_dict = {} 
        color_keys = list(json_opts.color_cls_boundary.keys())

        if args.run_func is None:
            # process res_df to demanded metrics
            # 1) loop thr colors, 2) util, 3) ncolor_found
            
            
            column_name_list = []
            for color_key in color_keys:
                column_name_list += [color_key+'_max']
            column_name_list += ['utility', 'ncolor_found']

            for col_nm in column_name_list:
                tmp_dict['ep0_'+col_nm] = eval_res_df[eval_res_df['epoch']==0][col_nm].values[0]
                tmp_dict['ep%d_'%total_epoch+col_nm] = eval_res_df[eval_res_df['epoch']==total_epoch][col_nm].values[0]
                tmp_dict['delta_'+col_nm] = tmp_dict['ep%d_'%total_epoch+col_nm] - tmp_dict['ep0_'+col_nm]
            # nexp to find all colors
            
            nepoch_find_allcolor = min(eval_res_df[eval_res_df['ncolor_found']==total_ncolors]['epoch'])
            nexp_find_allcolor = json_opts.model.init_training_size +  nepoch_find_allcolor * json_opts.model.batch_size
            tmp_dict['nexp_find_allcolor'] = nexp_find_allcolor
        else:
            if args.run_func == 'get_targetYield_vs_nexp':
                from utils.eval_MOO  import get_targetYield_vs_nexp
                tmp_dict = get_targetYield_vs_nexp(eval_res_df, init_tr_size=json_opts.model.init_training_size, 
                                            batch_size=json_opts.model.batch_size, colors=color_keys)
            elif args.run_func =='get_nexp_vs_utility':
                from utils.eval_MOO  import get_nexp_vs_utility
                tmp_dict = get_nexp_vs_utility(eval_res_df, init_tr_size=json_opts.model.init_training_size, 
                                            batch_size=json_opts.model.batch_size)
            else:
                raise Exception('Invalid evaluation funciton %s'%args.run_func)


        data = pd.DataFrame.from_dict(tmp_dict, orient='index').T
        print('final evaluation results....')
        print(data)
        data.index = [args.save_index]
        write_header = True if not os.path.exists(args.save_pth) else False
        data.to_csv(args.save_pth, mode='a',index=True, index_label='index',header=write_header)
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MOO experiment searcher')

    parser.add_argument('-c', '--config', default='./configs/debug_config_moo.json', help='training config file')#, required=True
    parser.add_argument('-d', '--debug',   help='returns runtime', action='store_true')
    parser.add_argument('-s', '--save_pth',  help='The csv path to save the result of the final epoch.')
    parser.add_argument('-i', '--save_index',  help='The index to save the result of the final epoch.')
    parser.add_argument('-f', '--run_func',  help='The function name to compute evaluation functions.')
    parser.add_argument('-m', '--mode', default='synthetic',  help='synthetic | real. Define whether it runs with a fully labelled synthetic dataset or in real mode')
    args = parser.parse_args()
 

    print('%s : Start running ...'%(get_cur_datetime()),flush=True)
    import time
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        run(args)
    print('time used: %.3f mins'%( (time.time()-start_time )/60))


