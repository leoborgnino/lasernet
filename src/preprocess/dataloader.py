from Waymo_pytorch_dataloader.waymo_pytorch_dataset import WaymoDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from six.moves import reduce
from six.moves import cPickle as pkl

class RangeImageLoader:

    def __init__(self):
        DATA_PATH = '../../datasets/waymo-dataset/'

        dataset = WaymoDataset(DATA_PATH, 'train', True, "new_waymo")

        frame, idx = dataset.data, dataset.count
        ri =  dataset.get_lidar(frame, idx)
        fig, ax = plt.subplots()        
        target = dataset.get_label(frame, idx)
        

sampling_rate = 1

# load dataset and prepare imdb for training
DATAPATH = '../../datasets/waymo-range/training/'
image_sets = os.listdir(DATAPATH)
print(image_sets)
roidbs = []
for image_set in image_sets:
    roidb = pkl.load(open(DATAPATH+image_set, "rb"), encoding="latin1")
    roidbs.append(roidb)
roidb = reduce(lambda x, y: x + y, roidbs)

# sampling rate
roidb = [r for idx, r in enumerate(roidb) if idx % sampling_rate == 0]

print(roidb)
            
clase = RangeImageLoader()

