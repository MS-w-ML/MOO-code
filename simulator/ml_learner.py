import os
import json
import pickle
import numpy as np
import pandas as pd 
import xgboost as xgb



if __name__=='__main__':
    import sys
    sys.path[0] = 'd:\\projects\\MOO'
from dataio.utils import load_XY
from utils.utils import recursive_mkdir
from models.base_model import get_ml_cv

class ML_Modelling():

    ## TODO: XGBoost, GPR, SVR, 
    # only cope with overlapping features in two datasets
    def __init__(self, ds_name, y_type, ml_model_name):

        self.y_type = y_type
        self.ds_name = ds_name
        self.ml_model_name = ml_model_name
        self.model_pth = './data/simulator/ml_learners/%s/%s/%s.sav'%(self.ds_name, self.y_type, self.ml_model_name)
        recursive_mkdir(os.path.dirname(self.model_pth))

        X, Y = load_XY(ds_name=ds_name, as_matrix=False, y_type=y_type, norm=True)
            
        # ML model 
        self.model = self.get_model(X, Y)

        print('Initialized ML_Modelling with dataset: %s and model: %s'%(ds_name, ml_model_name))
        

    
    def __call__(self, x):
        return self.model.predict(x)


    def get_model(self, X,y):  

        
        model_name = self.ml_model_name
        ds_name = self.ds_name

        model_pth = self.model_pth
        if os.path.exists(model_pth):
            model = pickle.load(open(model_pth,'rb')) # load models

        else: 
            
            model_cv = get_ml_cv(model_name, nsplit=5)
            model_cv.fit(X,y) # train model
            model = model_cv.best_estimator_
            pickle.dump(model, open(model_pth,'wb')) # save model to file
            print('Best score: %.3f'%model_cv.best_score_)
            print('Best params:',model_cv.best_params_)
            print('Saved the best model searched \'%s\' on dataset %s to %s'%(model_name, ds_name, model_pth ))
            with open( os.path.dirname(model_pth)+"/%s_tr_best_score.txt"%self.ml_model_name, 
                        "a") as f:
                f.write('%s  scores %.3f \n'%(model_pth, model_cv.best_score_))
             

        return  model




    
if __name__ == '__main__':
    ml_modelling = ML_Modelling(ds_name='cqd_raw', ml_model_name='XGBoost')
