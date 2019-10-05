import h5py

class GenartDataSet:
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.hf = h5py.File(h5_file, 'r')

    def __getitem__(self, idx):
        return self.hf["data"][idx]

    def __len__(self):
        return self.hf["data"].shape[0]

    def __call__(self):
        for i in range(len(self)):            
            yield self[i], self[i]

    @property
    def shape(self):
        return self.hf["data"].shape

    def __del__(self):
        self.hf.close()