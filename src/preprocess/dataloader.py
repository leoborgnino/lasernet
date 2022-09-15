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

        for j in range(50):
            frame, idx = dataset.data, dataset.count
            print(frame)
            exit()
            #calib = dataset.get_calib(frame, idx)
            ri =  dataset.get_lidar(frame, idx)
            #print(ri.shape)
            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            ax.imshow(ri[:,:,0], aspect='auto')

            #plt.imshow()
            #plt.show()
            target = dataset.get_label(frame, idx)
            #print(target)
            #print(i.box2d)

            # Create a Rectangle patch
            for i in target:
                print(i.box2d)
                print(i.cls_type)
                if(i.cls_type=='VEHICLE'):
                    ax.add_patch(patches.Rectangle((i.box2d[0], i.box2d[1]), i.box2d[2]-i.box2d[0], i.box2d[3]-i.box2d[1], linewidth=2.0, edgecolor='r', facecolor='r'))
                #if(i.cls_type=='SIGN'):
                #    ax.add_patch(patches.Rectangle((i.box2d[0], i.box2d[1]), i.box2d[2]-i.box2d[0], i.box2d[3]-i.box2d[1], linewidth=0.8, edgecolor='g', facecolor='none'))
            #plt.show()
            plt.savefig('%d'%j)
            plt.close()


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

