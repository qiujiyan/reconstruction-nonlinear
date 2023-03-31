import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
import numpy as np
from typing import *

from base import BaseVAE



class MLP(BaseVAE):
    def __init__(self,input_size:int, mod_number:int , pod_loss = None):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, mod_number)
        self.fc3 = nn.Linear(mod_number, 512)
        self.fc4 = nn.Linear(512, input_size)
        self.tanh =  nn.Tanh()
        if pod_loss == None :
            self.pod_loss = nn.MSELoss()
        else:
            self.pod_loss = pod_loss

    def encode(self,x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

    def decode(self,x):
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        return x

    def forward(self, x):
        self.code = self.encode(x)
        self.code_shape = self.code.shape
        self.out = self.decode(self.code)
        return self.out

    def pde_loss(self,x):
        x = x.reshape()
        loss_deg = nn.Conv2d()

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        input = args[1]
        loss = self.pod_loss(recons, input)

        return {'loss': loss, }
    
    def forward_loss(self,x,y):
        code = self.encoder(x)
        return self.decoder_loss(code,y)


