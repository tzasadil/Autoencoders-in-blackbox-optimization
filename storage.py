import numpy as np
import pandas as pd
import os
import glob
import numpy.ma as ma
import pd_cols
from doe2vec.doe2vec import doe_model 

DEFAULT_DATA_DIR = 'data'

# arr2str = lambda arr: ' '.join(str(x) for x in arr)
def arr2str(arr): 
    import numpy as np
    return np.array2string(arr,max_line_width=np.inf, separator=' ')[1:-1]
def str2arr(dtype):
    return lambda string: np.fromstring(string, dtype=dtype, sep=' ')
def listmap(func, collection):
    return list(map(func, collection))



def unzip(a):
    return list(map(list,list(zip(*a))))


def resolve_data_dir(data_dir=None):
    return data_dir or os.environ.get('BBOB_DATA_DIR', DEFAULT_DATA_DIR)


def store_data(df, desc='df', data_dir=None):
    if df.empty: return
    data_dir = resolve_data_dir(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    #ndarrays are stored separately as its way faster and takes less space
    # stored_separately = {}
    # stored_normally_cols = pd_cols.all_cols.copy()

    # for name in ['vals','evals','dists','correct_invariant']:
    #     stored_normally_cols.remove(name)
    #     arrs = df[name].to_list()
    #     lengths = np.stack([len(a) for a in arrs],axis=0)
    #     max_len = max(lengths)
    #     # aligned = np.stack([np.pad(a, (0,max_len - len(a)), mode='empty') for a in arrs],axis=0)
    #     aligned = [np.pad(a, (0,max_len - len(a)), mode='empty') for a in arrs]
    #     df[name] = aligned
    #     df[name+'_len'] = list(lengths)
    #     stored_separately[name]= aligned
    #     stored_separately[name+'_len']= lengths

    
    # df = df[stored_normally_cols] # get rid of augmentations and stored_separately cols
    df = df[pd_cols.all_cols] # get rid of augmentations and stored_separately cols
    # with pd.HDFStore(f'data/{desc}.h5','a') as data_storage:
    #     data_storage.put('df',df)
    np.savez_compressed(os.path.join(data_dir, desc), **{c: df[c].values for c in df.columns}, allow_pickle=True)
    # np.savez_compressed(f'data/{desc}', allow_pickle=True, **stored_separately)


def merge_and_load(data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    numpy_files = glob.glob(os.path.join(data_dir, '*.npz'))
    dfs = []
    for npz in numpy_files:
        data = np.load(npz, allow_pickle=True)
        df = pd.DataFrame({file: data[file] for file in data.files})
        dfs.append(df)
        data.close()
    if len(numpy_files)>0:
        del data   
    df = pd.concat(dfs, ignore_index=True) if len(dfs)>0 else None
    if len(numpy_files)>1:
        overwrite(df, data_dir=data_dir)
    return df
    

def overwrite(df, data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    df_path = os.path.join(data_dir, 'df.npz')
    if os.path.exists(df_path):
        os.remove(df_path)
    numpy_files = glob.glob(os.path.join(data_dir, '*.npz'))
    store_data(df, data_dir=data_dir)
    for f in numpy_files:
        if os.path.normpath(f) == os.path.normpath(df_path):
            continue
        os.remove(f)


def load_data(data_dir=None):
    return merge_and_load(data_dir=data_dir)
    
