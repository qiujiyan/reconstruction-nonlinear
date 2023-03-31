
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection
import einops
class FluidFieldDataset(data.Dataset):
    def __init__(self,data,):
        from os import path
        self.data = data
        self.len  = data.shape[0]
        self.shape = data.shape[1:]

    def getFildSize(self):
        return len(self.data[0])

    def __len__(self,):
        return self.len

    def __getitem__(self,i):
        return torch.tensor(self.data[i],dtype=torch.float32)


def build_dataset(dataset_path,dataset_type,test_size,):
    

    np.random.seed(42)

    if 'cylinder2d-p'.lower() == dataset_type.lower():
        d = np.load(dataset_path)
        d = np.nan_to_num(d,0)
        d = d[:,:,:]
        np.random.shuffle(d)
        test_num = int(test_size*len(d))
        data_shape = d.shape[1:]
        t = d[test_num:]
        v = d[:test_num]
        res = {
            'data_shape':data_shape,
            'train_dataset':FluidFieldDataset(t),
            'val_dataset':FluidFieldDataset(v)
        }
        return res
    if 'cylinder2d-puv'.lower() == dataset_type.lower():
        d_p = np.load(dataset_path['p'])
        d_u = np.load(dataset_path['u'])
        d_v = np.load(dataset_path['v'])
        d = np.array([d_p,d_u,d_v]).transpose((0,1,2,3))
        d = einops.rearrange(d,'c b h w -> b c h w')
        d = np.nan_to_num(d,0)
        d = d[:,:,:]
        np.random.shuffle(d)
        test_num = int(test_size*len(d))
        data_shape = d.shape[1:]
        t = d[test_num:]
        v = d[:test_num]
        res = {
            'data_shape':data_shape,
            'train_dataset':FluidFieldDataset(t),
            'val_dataset':FluidFieldDataset(v)
        }
        return res
    if 'cylinder2d-p-256'.lower() == dataset_type.lower():
        d = np.load(dataset_path)
        d = np.nan_to_num(d,0)
        d = d[:,:256,0+100:256+100]
        np.random.shuffle(d)
        test_num = int(test_size*len(d))
        data_shape = d.shape[1:]
        t = d[test_num:]
        v = d[:test_num]
        res = {
            'data_shape':data_shape,
            'train_dataset':FluidFieldDataset(t),
            'val_dataset':FluidFieldDataset(v)
        }
        return res
    
    if 'cylinder2d-puv-256'.lower() == dataset_type.lower():
        d_p = np.load(dataset_path['p'])
        d_u = np.load(dataset_path['u'])
        d_v = np.load(dataset_path['v'])
        d = np.array([d_p,d_u,d_v])
        d = einops.rearrange(d,'c b h w -> b c h w')
        d = np.nan_to_num(d,0)
        d = d[:,:256,0+100:256+100]
        np.random.shuffle(d)

        test_num = int(test_size*len(d))
        data_shape = d.shape[1:]
        t = d[test_num:]
        v = d[:test_num]
        res = {
            'data_shape':data_shape,
            'train_dataset':FluidFieldDataset(t),
            'val_dataset':FluidFieldDataset(v)
        }
        return res