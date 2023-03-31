import torch
from typing import *

from base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import numpy as np


class CNNSEA(BaseVAE):
    def __init__(self,
                 mod_input_shape,
                 latent_dim: int,
                 podloss,
                 **kwargs) -> None:
        super(CNNSEA, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        hidden_dims = [32, 64, ]
        self.ft_size_x = int(mod_input_shape[0]/4)
        self.ft_size_y = int(mod_input_shape[1]/4)

        self.hidden_dims = hidden_dims[:]
        # Build Encoder
        in_channels = 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 11, stride= 2, padding  = 5),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_output =  nn.Sequential(
            nn.Linear(hidden_dims[-1]*self.ft_size_x*self.ft_size_y, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )


        # Build Decoder
        modules = []

        self.decoder_input =  nn.Sequential(
            nn.Linear(latent_dim , 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dims[-1]*self.ft_size_x*self.ft_size_y),
            nn.ReLU(),        
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=11,
                                       stride = 2,
                                       padding=5,
                                       output_padding=1),
                      nn.Tanh(),
                )
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.Tanh(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1, kernel_size= 3, padding= 1)
                            )

    def encode(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        code = self.encoder_output(result)
        return code

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        result = z.view(-1, self.hidden_dims[-1], self.ft_size_x, self.ft_size_y)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        code = self.encode(input)
        return  self.decode(code)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        if (recons.shape != input.shape):
            recons = recons.flatten()
            input = input.flatten()
            
        loss = self.pod_loss(recons, input)

        return {'loss': loss, }

