from .synthetic_dataset import SO_SyntheticDataset, MOO_SyntheticDataset



def get_ds(ds_type):
    
    if ds_type == 'SO':
        return SO_SyntheticDataset
    elif ds_type == 'MOO':
        return MOO_SyntheticDataset
    else:
        raise Exception('Invalid dataset name: %s'%ds_type) 
    


    


            







