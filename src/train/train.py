import torch
import sys 
sys.path.append('../model')
sys.path.append('../preprocess')

from LaserNet import LaserNet
from Waymo_pytorch_dataloader.waymo_pytorch_dataset import WaymoDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

DATA_PATH = '../../datasets/waymo-dataset/'
training_data = WaymoDataset(DATA_PATH, 'train', True, "new_waymo")
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
img = train_features[0].squeeze()

#fig, ax = plt.subplots()
#ax.imshow(img[:,:,0], aspect='auto')
#plt.show()


print(train_features.shape)
train_features = train_features.permute(0, 3, 1, 2) # from NHWC to NCHW
print(train_features.shape)
model = LaserNet(train_features.shape[1],64)
model.float()
model(train_features.float())
