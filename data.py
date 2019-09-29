import torch
import h5py

class GanartDataSet(torch.utils.data.Dataset):
    def __init__(self, h5_file, key='data'):
        super().__init__()

        self.h5_file = h5_file
        self.h5f = h5py.File(h5_file, 'r')
        self.ds = self.h5f[key]

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, index):
        
        return self.ds[index], 0

    def __del__(self):
        self.h5f.close()
        
            
        
