import h5py

def load_data(fname):    
    with h5py.File(fname,'r') as f:
        data = f['data'][:]

    return data