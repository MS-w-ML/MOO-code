import os
import shutil
from datetime import date
from datetime import datetime


def normalize(x, old_max, old_min, new_max, new_min):
    '''Min-max ormalization'''
    x_new = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return x_new


def convert_s2hr(secs):
    '''Convert seconds to hours, e.g. 600 s -> 0 hrs 10 mins'''

    mins = secs / 60 
    hrs = int(mins / 60)
    return '%d hrs %d mins' %(hrs, mins%60)


def get_cur_datetime():
    '''Return current date and time in string'''

    current_time = datetime.now().strftime("%H:%M:%S")
    today = date.today()

    return " %s / %s "%(today, current_time)




def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def recursive_mkdir(dir):
    '''Recursively mkdir. E.g. dir=/a/b/c/d, if only /a/b exists, /a/b/c and /a/b/c/d/ will be created.'''

    if os.path.exists(dir):
        return 

    path_splitted = os.path.split(os.path.abspath(dir))
    print('recursive_mkdir',path_splitted)

    if not os.path.exists(path_splitted[0]):
        recursive_mkdir(path_splitted[0])
    
    mkdir(dir)
    print('Create dir: %s'%dir)




def mkdir(path, overwrite = False, verbose=True):
    '''If path not exits, create path;
    If the path exists but $overwrite$ is True, delete existing directory and create a new folder;
    otherwise, issue an warning
    Return:
        created: Bool. Whether the path is created.
    '''

    created = False
    if not os.path.exists(path): 
        os.makedirs(path) # recurvely mkdirs
        msg = 'Created new dir: '+path
        created = True
    elif overwrite:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        msg = 'Overwrite dir: '+path
        created = True
    else:
        msg = "Path: %s exists." % path

    if verbose: print(msg)
    return created

    

def mkdirs(paths, overwrite = False, verbose=True):
    '''Calls mkdir().
    
    Return:
        created: list of Bool. Each corresponds to whether the path is created.
    
    '''
    if not isinstance(paths, Iterable):
        paths = [paths]

    created = []
    for path in paths:
            created.append(mkdir(path, overwrite, verbose))

    return created
