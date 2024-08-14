import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from diffusion.utils import DeviceDataLoader

class Dataset1D(Dataset):
    def __init__(self, data_path, subset=False):
        super().__init__()
        # get data
        self.df = pd.read_csv(data_path, header=None)
        if subset:
            self.df = self.df[:500]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get data and convert to numpy
        data = self.df.iloc[idx].values.astype(np.float32)
        # convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        # transform to [-1,1]
        data_tensor = (data_tensor * 2) - 1
        # add dimension for channel and return
        return data_tensor.unsqueeze(0)

def get_dataset(dataset_name):
    name = dataset_name.lower()
    path = str(f'data\embeddings\\top25_104\{name}.csv')
    dataset = Dataset1D(data_path=path)
    
    return dataset

def get_dataloader(dataset_name='MNIST',
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"
                  ):
    dataset    = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle
                           )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 1.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) #* 255.0