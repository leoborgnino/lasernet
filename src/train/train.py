import torch
import sys 
sys.path.append('../model')

from LaserNet import LaserNet

model = LaserNet(10,25)
print(model)
