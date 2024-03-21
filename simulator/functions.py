'''Function generators'''
import os
import joblib
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from utils.utils import recursive_mkdir
from dataio.utils import load_XY 

def get_function_class(name):
        if name.upper() == 'POLYFIT':
            return POLYFIT
        elif name.upper() == 'RANDOM':
            return RandomGenerator
        else:
            raise NotImplementedError('This synthetic function is not defined: %s'%name)


class POLYFIT():
    '''Sklearn polynomial fitting on given data'''
    def __init__(self, ds_name, deg, overwrite=False, y_type=None):
        self.ds_name  = ds_name
        self.deg = deg
        cls_nm = self.__class__.__name__
        self.pth = './data/simulator/functions/%s/%s/deg%d.joblib'%(cls_nm, self.ds_name, deg)
        to_dir = os.path.dirname(self.pth)
        recursive_mkdir(to_dir)
    

        if os.path.exists(self.pth) and not overwrite:
            self.polyreg_scaled = joblib.load(self.pth)
            print('Loaded existing functions.POLYFIT from %s'%self.pth)
            
        else:
            self.polyfit(y_type)
            joblib.dump(self.polyreg_scaled, self.pth) 
            print('Saving new functions.POLYFIT to %s'%self.pth)
           
        print('Initialized functions.POLYFIT with dataset: %s and deg=%d'%(ds_name, deg))
        
    def __call__(self, x):
        return self.polyreg_scaled.predict(x)



    def polyfit(self, y_type):
        X, Y = load_XY(ds_name=self.ds_name, as_matrix=False, y_type=y_type)
        self.polyreg_scaled = make_pipeline(PolynomialFeatures(self.deg), StandardScaler(), LinearRegression())
        self.polyreg_scaled.fit(X, Y)




class RandomGenerator(): #TODO: save 
    def __init__(self, nsample=0, name='', overwrite=False):
        self.nsample = nsample
        cls_nm = self.__class__.__name__
        self.pth = './data/simulator/functions/%s/randomDS_%s_ns%d'%(cls_nm, name, self.nsample)+'.npy'

        if overwrite or not os.path.exists(self.pth):
            print('Saving new random dataset to %s'%self.pth)
            with open(self.pth, 'wb') as f:
                self.y =  np.random.rand(self.nsample) #* 2 - 1 
                np.save(f, self.y)
        else:
            print('Loading existing random dataset from %s'%self.pth)
            with open(self.pth, 'rb') as f:
                self.y = np.load(f)
                
    
    def __call__(self, x):
        index = np.array(x.index).astype(int)
        return self.y[index]

