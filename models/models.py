from MLP import MLP
from CNNSEA import CNNSEA
from typing import *

from functools import reduce


def build_models(model_type : str ,mod_input_shape  ,mod_number :int ,pod_loss=None):
    if model_type== 'MLP':
        MLP_input_size = reduce(lambda x,y:x*y,mod_input_shape)
        return {
            'mod_input_shape':(MLP_input_size,),
            'mod_output_shape':(MLP_input_size,),
            'code_shape':(mod_number,),
            'model':MLP(MLP_input_size,mod_number,pod_loss)
        }
    if model_type== 'CNNSEA':

        return {
            'mod_input_shape':[1,]+list(mod_input_shape),
            'mod_output_shape':[1,]+list(mod_input_shape),
            'code_shape':(mod_number,),
            'model':CNNSEA(mod_input_shape,mod_number,pod_loss)
        }
        
    