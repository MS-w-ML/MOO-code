from .pam_model import PAM_Model_SO, PAM_Model_MOO


def get_model(opt):
    model_name = opt.model.name
    ds_type = opt.dataset.type

    if ds_type == 'SO':
        model_dict = {'PAM':PAM_Model_SO,
                       }
        
    elif ds_type == 'MOO':
        model_dict = {'PAM':PAM_Model_MOO,
                       }
    else:
        raise Exception('Invalid model name: %s'%model_name) 


    return model_dict[model_name]




