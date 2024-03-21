import pandas

def get_targetYield_vs_nexp(eval_res_df, init_tr_size, batch_size, colors):
    epochs = eval_res_df['epoch']
    target_yield_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    _target_yield_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    tmp_dict = {}
    for target_yield in target_yield_list:
        tmp_dict['nexp_target%d'%int(target_yield*100)] = -1


    for epoch in epochs:
        df_tmp = eval_res_df[eval_res_df['epoch']==epoch]
        for target_yield in target_yield_list:
            allcolor_lt_target = True
            for color in colors:
                allcolor_lt_target = allcolor_lt_target and (df_tmp[color+'_max'].values[0] >= target_yield)
                if not allcolor_lt_target:
                    break
            if allcolor_lt_target:
                tmp_dict['nexp_target%d'%int(target_yield*100)] = init_tr_size + batch_size * epoch
                _target_yield_list.remove(target_yield)
        target_yield_list = _target_yield_list.copy()  
        if len(target_yield_list) == 0:
            break   

    return tmp_dict




def get_nexp_vs_utility(eval_res_df, init_tr_size, batch_size):
    epochs = [5,10,15,20,25,30,35,40,45,50]
    tmp_dict = {}
    for epoch in epochs:
        nexp = init_tr_size + batch_size * epoch
        tmp_dict['util_nexp%d'%int(nexp)] = eval_res_df[eval_res_df['epoch']==epoch]['utility'].values[0]
    return tmp_dict