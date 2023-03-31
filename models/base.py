
from typing import *
import torch
from torch import nn
from torch import Tensor
import numpy as np
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        self.pod_loss = nn.MSELoss()

    def encode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


    def gappy_decoder_loss(self, code, y ,mask_map):
        per = self.decode(code)
        per = mask_map*per
        y = mask_map*y
        if (per.shape != y.shape):
            per = per.flatten()
            y = y.flatten()
            
        loss = self.pod_loss(per,y)*1e6
        return loss

    def loss_decoder_helper(self,fix_y,mask_map,device='cuda'):
        def loss_decoder(input_x):
            # if len(input_x.shape)!=2: 
            input_x =  torch.Tensor(input_x.astype(np.float32)).to(device)

            mask_map_t = mask_map.to(device)
            return self.gappy_decoder_loss(input_x,fix_y.to(device),mask_map_t).detach().cpu().numpy()
        return loss_decoder


    def grad_loss_decoder_helper(self,y,mask_map,device='cuda'):
        def fp(input):
            def fn(x,y,m):
                return self.gappy_decoder_loss(x,y,m)

            input =  torch.Tensor(input.astype(np.float32)).to(device)

            mask_map_t = mask_map.to(device)

            input.requires_grad_()
            y.requires_grad_()
            mask_map_t.requires_grad_()

            gradient = torch.autograd.grad(outputs=self.gappy_decoder_loss(input,y.to(device),mask_map_t), inputs = input)[0]
            return gradient.detach().cpu().numpy().flatten()
        return fp
        
