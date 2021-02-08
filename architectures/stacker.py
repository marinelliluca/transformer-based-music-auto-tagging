import torch
import torch.nn as nn
import torch.nn.functional as F
from .front.conv import BlockChoi

"""
Example:

stack_dict = {"list_out_channels":[64,128,128,256,256,646], 
              "list_kernel_sizes":[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
              "list_pool_sizes":  [(4,1),(2,2),(2,2),(2,2),(2,2),(2,2)], 
              "list_avgpool_flags":[False,False,False,False,False,True]}

conv_stack = ConvStack(stack_dict)

print(conv_stack)

print(sum(p.numel() for p in conv_stack.parameters() if p.requires_grad)) # number of trainable parameters

print(conv_stack(torch.rand((32,1,128,646))).shape)
"""

class ConvStack(nn.Module):
    
    def __init__(self,stack_dict,in_channels=1,version="choi"):
        super(ConvStack, self).__init__()
        
        self.depth = len(stack_dict["list_out_channels"])
        self.version = version
        
        if version=="choi":
            # set class attributes in a for loop
            for i in range(self.depth):
                setattr(self, 
                        f"conv_block_{i}", 
                        BlockChoi(in_channels if i==0 else stack_dict["list_out_channels"][i-1],
                                  stack_dict["list_out_channels"][i],
                                  stack_dict["list_kernel_sizes"][i],
                                  stack_dict["list_pool_sizes"][i],
                                  stack_dict["list_avgpool_flags"][i]))
    
    def forward(self, inputs):
        
        if self.version=="choi":
            x = getattr(self,f"conv_block_{0}")(inputs)

            for i in range(1,self.depth):
                x = getattr(self,f"conv_block_{i}")(x)

            # squeeze the singleton frequency dimension
            if x.size()[2]==1: 
                x = x.squeeze(2)
            else:
                raise Exception("Insufficient pooling along the frequency axis, "+
                                "the required resulting size is 1.")    
        return x
            
