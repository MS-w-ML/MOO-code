'''
All evaluation metrics:

- prediction: r2, mse, pearson (+ p value)

- ranking: MAP, NDCG
'''
import numpy as np
import pandas as pd 
from sklearn import metrics 
from scipy.stats import pearsonr, rankdata
from utils.utils import normalize

def compute_prediction_metric(y_true, y_pred):
    '''Compute regression metrics : r2, mse, pearson (+ p value)
    
    return dict of 4 metrics
    '''
    # compute prediction metrics
    r2 = metrics.r2_score(y_true,y_pred)   
    mse = metrics.mean_squared_error(y_true,y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    if len(y_true) > 1:
        [pear,p_value] = pearsonr(y_true,y_pred)
    else:
        r2, pear, p_value = None, None, None
    return {'r2':r2,'mse':mse,'mape':mape,'pearson':pear,'p_val':p_value}


def compute_rank_metrics(y_true, y_pred, k=10, fillna=False):
    '''Compute rank metrics : 
    1) MAP-related (# = max_k+2): AP@k=1,...,max_k, mean of [AP@k=1,...,max_k], MAP
    2) NDCG-related (# = max_k): NDCG@k=1,...,max_k
    return dict 
    '''
    max_k = min(y_true.shape[0], k)
    df = pd.DataFrame(np.concatenate((y_true.reshape(-1,1),y_pred.reshape(-1,1)), axis=1), columns=['y_true', 'y_pred'])
    
    # convert to rank
    df['y_true_rank'] = convert_real_to_rank(df['y_true'], ascending=True)
    df['y_pred_rank'] = convert_real_to_rank(df['y_pred'], ascending=True)
    df = df.sort_values('y_true_rank')

    # compute MAP-related
    res_dict = {}
    precision_dict = compute_hit_ratio(df['y_pred_rank'].to_numpy())
    precision_sum = 0

    
    for i in range(1, max_k+1):
        p_k = precision_dict['hit_rate@k=%d'%i]
        res_dict['precision@k=%d'%i] = p_k
        precision_sum += p_k
    res_dict['precision_all_mean'] = precision_dict['hit_rate_mean']
    res_dict['precision_k_mean'] = precision_sum / max_k 
    
    # compute NDCG-related
    dcg_res_dict = compute_norm_DCG(df['y_pred_rank'], df['y_true_rank'], k=max_k, norm_idcg=True)

    res_dict.update(dcg_res_dict)

    if fillna and max_k < k:
        for i in range(max_k+1, k+1):
            res_dict['precision@k=%d'%i] = None
            res_dict['NDCG@k=%d'%i] = None


    return res_dict



def convert_real_to_rank(vals, ascending=True):
    '''
    Convert real values to rank. If ascending, then larger val lead to better rank,
    e.g. If ascending: [0.1, 0.5] -> [2, 1]
         if descending: [0.1, 0.5] -> [1, 2]

    Rank 1 is better than rank 2
    '''

    if ascending:
        vals = -vals

    ranks = rankdata(vals, method='min')#, nan_policy='raise'
    return ranks


def compute_hit_ratio(y_pred_rank, max_k=None):

    if max_k is None:
        max_k = y_pred_rank.shape[0]

    res_dict = {}
    hit_rate_sum = 0
    for i in range(max_k):
        top_k_mask = y_pred_rank[:i+1] <= i+1
        hit_rate = top_k_mask.sum() / (i+1)
        res_dict['hit_rate@k=%d'%(i+1)] = hit_rate
        hit_rate_sum += hit_rate
    res_dict['hit_rate_mean']  = hit_rate_sum / max_k

    return res_dict



def compute_norm_DCG(y_pred_rank, y_true_rank, k, norm_idcg=True):
    '''Compute normalized discounted cumulative gain for rankings
    
    return dict of k DCG (for @1,..,k)
    '''
    
    # normalize ranks
    y_true_rank_norm = normalize(x=y_true_rank, old_max=max(y_true_rank), old_min=min(y_true_rank), new_max=1, new_min=0) # norm to [0, 1] for reducing computation overheads
    y_pred_rank_norm = normalize(x=y_pred_rank, old_max=max(y_pred_rank), old_min=min(y_pred_rank), new_max=1, new_min=0) # norm to [0, 1] for reducing computation overheads
    
    # Dk and Gk
    g_k = np.power(2, 1-y_pred_rank_norm) 
    d_k_inv = np.log2(np.arange(1, y_pred_rank_norm.shape[0]+1) + 1) 
    dg = g_k / d_k_inv
    idcg = 1

    if norm_idcg:
        # IDk and IGk
        ig_k = np.power(2, 1-y_true_rank_norm) 
        id_k_inv = np.log2(np.arange(1, y_true_rank_norm.shape[0]+1) + 1) 
        idg = ig_k / id_k_inv
        idcg = 0

    dcg_res_dict = {}
    dcg = 0
    
    for i in range(k):
        dcg += dg[i]
        if normalize:
            idcg += idg[i]
        dcg_res_dict['NDCG@k=%d'%(i+1)] = dcg / idcg

    return dcg_res_dict







