
import argparse
import os, time, random
import pandas as pd

from dataio.dataset import SyntheticDataset
from dataio.utils import mkdir, mkdirs, append_csv
from models.PAM_regression import PAM_regression
from models.adaptive_design import adaptive_design

def evaluate_SO(args):
    '''Evaluate search algorithm based on synthetic f: X -> Y
    '''

    ## generate synthetic dataset
    ds = SyntheticDataset(func_name=args.function_name)

    ## perform algorithm evaluation
    # build directory
    model_name = args.model
    root_dir = './results/'+args.exp_name
    result_dir = os.path.join(root_dir,args.function_name)
    mkdir([root_dir, result_dir], overwrite=False, ignore=True)
    model_result_dir = os.path.join(result_dir ,model_name)
    mkdirs([ model_result_dir], args.overwrite, ignore=True)

    init_train_size = args.init_train_size
    step_size = args.step_size
    # create initial training set
    full_set_indexes = list(range(1, len(ds)+1))
    #full_set_indexes_wo_best.remove(max_ind)
    print('Evaluate model %s with step size %d and initial training size %d'%(model_name,step_size, init_train_size), flush=True)

    # results store
    res_csv_title = model_name+'_'+args.function_name
    all_res_csv_path_nc = os.path.join(model_result_dir, 'ALL_results_eval_SO.csv')
    total_time = 0
 
    for loop_count in range(args.nloop):
        
        # check number of recorded loop
        if os.path.exists(all_res_csv_path_nc) and pd.read_csv(all_res_csv_path_nc).shape[0] >= args.nloop:
            break
        init_time = time.time()
        # create initial training set
        init_train_set_indexes = random.sample(full_set_indexes, k=init_train_size)

        if model_name == 'PAM':
            result = PAM_regression(ds, inner_nsplits=10, init_train_indexes = init_train_set_indexes, save_csv= True, verbose=args.debug, patience=10, batch = step_size, title=res_csv_title,save_dir= model_result_dir)
        else:
            result = adaptive_design(ds, inner_nsplits=10, init_train_indexes = init_train_set_indexes, save_csv= True, verbose=args.debug, patience=10, batch = step_size, title=res_csv_title,save_dir= model_result_dir)

        # results storing
        res_dict = {'title':result['save_title'], 'Nc':result['Nc'], 'Nc_notBatch': result['Nc_notbatch']}
        append_csv(all_res_csv_path_nc, res_dict)

        iter_time = ( time.time() - init_time ) / 60 
        total_time += iter_time
        print('[%d / %d]'%(loop_count,args.nloop ),' -> ',str(res_dict),' time=  %.2f / total: %.2f mins'%(iter_time,total_time), flush=True)

    print('Finished! Saved results to %s'%model_result_dir, flush=True)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adaptive guidance for MS synthesis')

    parser.add_argument('-c', '--exp_name', required=True, help='Experiment name')
    parser.add_argument('-m','--model', required=True, help='Model name. < PAM | Adaptive_Design>')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Whether to overwrite existing checkpoints')
    parser.add_argument('-n', '--nloop', type=int,  default=1, help='Number of loops to perform')
    parser.add_argument('-d', '--debug', action='store_true', help='Whether in debug mode')
    parser.add_argument('-f', '--function_name', default='linear',  help='The function to be called')
    parser.add_argument('-s', '--step_size',type=int,  default=1,  help='step size for function')
    parser.add_argument('-tr', '--init_train_size',type=int,  default=10,  help='initial training set size')
    args = parser.parse_args()

    assert args.model in ['PAM', 'Adaptive_Design'], 'Invalid Model type: %s'%args.model
    assert args.function_name in ['linear', 'quadratic', 'cubic'], 'Invalid function name input by user: %s'%args.function_name
    evaluate_SO(args)

