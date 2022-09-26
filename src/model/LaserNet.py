import torch
from torch import nn
import torch.nn.functional as F


class LaserNet(nn.Module):

    def __init__(self, num_inputs, num_filters):#, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.block_1a = FeatureExtractor(num_inputs, num_filters, downsample=False, reshape=True)
        self.block_1b = FeatureAggregator(num_filters, num_filters)
        self.block_2a = FeatureExtractor(num_filters, num_filters)
        self.block_1c = FeatureAggregator(num_filters, num_filters)
        self.block_2b = FeatureAggregator(num_filters, num_filters)
        self.block_3a = FeatureExtractor(num_filters, num_filters)
        
    def forward(self, x):
        print(x.shape)
        extract_1 = self.block_1a(x)
        print(extract_1.shape)
        extract_2 = self.block_2a(extract_1)
        extract_3 = self.block_3a(extract_2)
        aggregate_1 = self.block_1b(extract_1)
        aggregate_2 = self.block_2b(extract_2)
        
        return x

class ResnetBlock(nn.Module):

    def __init__(self, in_channels,out_channels, kernel_size=(3,3), stride=(1,1), reshape=False, name=None):
        super().__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        print(x.shape)
        out = self.block(x)
        
        if self.skip is not None:
            print("IDENTITY")
            print(x.shape)
            identity = self.skip(x)
            print(identity.shape)

        print(out.shape)
        print(identity.shape)
        out += identity
        out = F.relu(out)

        return out

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels, num_filters, num_blocks=7, downsample=True, reshape=False, name='FeatureExtractor'):
        super().__init__()

        self.sequence = []
        # Downsample by 2 along horizontal
        if downsample:
            self.sequence.append(ResnetBlock(input_channels,num_filters, stride=(1, 2), name="Downsample", reshape=True))
        else:
            self.sequence.append(ResnetBlock(input_channels,num_filters, name="Downsample", reshape=True))
        for i in range(num_blocks-1):
            self.sequence.append(ResnetBlock(num_filters,num_filters, name="Resnet_%i" % i, reshape=False))

        self.sequence = nn.Sequential(*self.sequence)

    def forward(self, x):
        return self.sequence(x)

class FeatureAggregator(nn.Module):
    def __init__(self, input_channels,num_filters, name='FeatureAggregator'):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=input_channels,out_channels=num_filters, kernel_size=(3, 3), stride=(1,2))
        self.bn = nn.BatchNorm2d(num_filters)
        self.block1 = ResnetBlock(num_filters,num_filters)
        self.block2 = ResnetBlock(num_filters,num_filters)

    def forward(self, fine_input, coarse_input):
        x = self.upsample(coarse_input)
        x = self.bn(x)
        x = nn.ReLU()(x)
        y = nn.concat([fine_input, x])
        y = self.block1(y)
        return self.block2(y)
