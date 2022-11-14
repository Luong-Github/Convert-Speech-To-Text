import torch.nn as nn
from torch import Tensor

class Swich(nn.Module):
    """
    Swich is a smooth, non-monitanic function

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(Swich, self).__init__()
        
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()
    
class GLU(nn.Module):
    # Gated Linear Units
    # dimension - dim 
    def __init__(self, dim: int):
        super(GLU, self).__init__()
        self.dim = dim
    
    # Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
    # sigmoid 
    # input -> arange of int -> chunk = 2 with dimension = self.dim
    def forward(self, input: Tensor) -> Tensor:
        outputs, gate = input.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    