import torch
from torch import nn
import torch.nn.functional as F


class LaserNet(nn.Module):

    def __init__(self, num_inputs, num_filters):#, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.block_1a = FeatureExtractor(num_inputs, num_filters, downsample=False, reshape=True)
        self.block_1b = FeatureAggregator(num_inputs, num_filters)
        self.block_2a = FeatureExtractor(num_inputs, num_filters)
        self.block_1c = FeatureAggregator(num_inputs, num_filters)
        self.block_2b = FeatureAggregator(num_inputs, num_filters)
        self.block_3a = FeatureExtractor(num_inputs, num_filters)
        
    def forward(self, x):
        extract_1 = self.block_1a(x)
        extract_2 = self.block_2a(extract_1)
        extract_3 = self.block_3a(extract_2)
        aggregate_1 = self.block_1b(extract_1)
        aggregate_2 = self.block_2b(extract_2)
        
        return x

class ResnetBlock(nn.Module):

    def __init__(self, input_channels,num_filters, kernel_size=(3,3), stride=(1,1), reshape=False, name=None, padding='same'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding,bias=False)

        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

        if reshape or stride != (1, 1):
            # Need to match skip connection dimensions to new dimensions
            self.conv3 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters, kernel_size=(1, 1), stride=stride,bias=False)
            self.skip_conn = lambda x, **kwargs: self.conv3(x)
            # self.skip_conn.add()
            # self.skip_conn.add()
        else:
            # Do nothing
            self.skip_conn = lambda x, **kwargs: x

    def forward(self, x):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = nn.ReLU(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        residual = self.skip_conn(input_tensor, training=training)
        x += residual
        return nn.ReLU(x)


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels,num_filters, num_blocks=7, downsample=True, reshape=False, name='FeatureExtractor'):
        super().__init__()

        self.sequence = []
        # Downsample by 2 along horizontal
        if downsample:
            self.sequence.append(ResnetBlock(input_channels,num_filters=num_filters, stride=(1, 2), name="Downsample", reshape=True, padding='valid'))
        else:
            self.sequence.append(ResnetBlock(input_channels,num_filters=num_filters, name="Downsample", reshape=True))
        for i in range(num_blocks-1):
            self.sequence.append(ResnetBlock(input_channels,num_filters=num_filters, name="Resnet_%i" % i, reshape=False))

        self.sequence = nn.Sequential(*self.sequence)

    def forward(self, input_tensor, training=False):
        return self.sequence(input_tensor, training=training)

class FeatureAggregator(nn.Module):
    def __init__(self, input_channels,num_filters, name='FeatureAggregator'):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=input_channels,out_channels=num_filters, kernel_size=(3, 3), stride=(1,2), padding='same')
        self.bn = nn.BatchNorm2d(num_filters)
        self.block1 = ResnetBlock(num_filters,num_filters)
        self.block2 = ResnetBlock(num_filters,num_filters)

    def forward(self, fine_input, coarse_input, training=False):
        x = self.upsample(coarse_input)
        x = self.bn(x, training=training)
        x = nn.ReLU(x)
        y = nn.concat([fine_input, x])
        y = self.block1(y, training=training)
        return self.block2(y, training=training)
