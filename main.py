import torch
import torch.nn as nn
import torch.functional as F
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

a=torch.arange(2*3*4*5).view(2,3,4,5).transpose(-2,-1)
b=torch.arange(2*3*4*5).view(2,3,4,5)
c=torch.matmul(b,a)
print(c.shape)