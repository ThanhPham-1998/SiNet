import torch 
import torch.nn as nn 

################################################################
#                                                              #
#    - Define  conv block                                      #
#                                                              #
# ##############################################################


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d((1,1,1,1))
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.zero_pad(x)
        x = self.conv_1(x)
        x = self.norm(x)
        return self.relu(x)


################################################################
#                                                              #
#    - Define  S2 Module                                       #
#                                                              #
# ##############################################################


"""
    - Channel shuffle 
"""
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


"""
    - Deepthwise 
        + in_channels: number channel of features before foward into module
        + kernel_per_layer: the number of kernels use for every channel
        + out_channels: output channels 
"""
class Deepthwise(nn.Module):
    def __init__(self, in_channels, kernel_per_layer, kernel_size, padding=1):
        super().__init__()
        self.deepthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * kernel_per_layer,
                                kernel_size=kernel_size, groups=in_channels, padding=padding)
    def forward(self, x):
        return self.deepthwise(x)

"""
    - Pointwise
"""
class Pointwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1)
    def forward(self, x):
        return self.pointwise(x)


"""
    - DeepthwiseSeparableConv
"""


class DeepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_per_layer, kernel_size, padding=1):
        super().__init__()
        self.deepthwise = Deepthwise(in_channels=in_channels, kernel_per_layer = kernel_per_layer,
                                    padding=padding, kernel_size=kernel_per_layer)
        self.norm = nn.BatchNorm2d(num_features=in_channels * kernel_per_layer)
        self.prelu = nn.PReLU()
        self.pointwise = Pointwise(in_channels=kernel_per_layer * in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.deepthwise(x)
        x = self.norm(x)
        x = self.prelu(x)
        x = self.pointwise(x)
        return x

"""
    - S2Block
"""
class S2Block(nn.Module):

    def __init__(self, reduce_ratio, in_channels, out_channels, kernel_size, padding=1, kernel_per_layer=1):
        super().__init__()
        self.avgpooling2d = nn.AvgPool2d(kernel_size=reduce_ratio, stride=reduce_ratio)
        self.separabble_conv = DeepthwiseSeparableConv(in_channels = in_channels,
                                                        out_channels=out_channels,
                                                        kernel_per_layer = kernel_per_layer,
                                                        kernel_size = kernel_size,
                                                        padding=padding
                                                    )
        self.up_sampling = nn.UpsamplingBilinear2d(scale_factor=reduce_ratio)
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.avgpooling2d(x)
        x = self.separabble_conv(x)
        x = self.up_sampling(x)
        x = self.norm(x)
        return x

"""
    S2Module
"""
class S2Module(nn.Module):
    def __init__(self, n_groups, reduce_ratio, in_channels_conv, kernel_per_layer,
                kernel_size_1, kernel_size_2, padding_1, padding_2):
        super().__init__()
        in_channels = in_channels_conv // 2
        out_channels = in_channels_conv // 2
        self.channel_shuffle = ChannelShuffle(groups = n_groups)
        self.pointwise = Pointwise(in_channels=in_channels_conv, out_channels=out_channels)
        self.s2block_1 = S2Block(reduce_ratio=reduce_ratio, in_channels= in_channels, 
                                    out_channels=out_channels, kernel_per_layer=kernel_per_layer, 
                                    kernel_size=kernel_size_1, padding=padding_1)
        self.s2block_2 = S2Block(reduce_ratio=reduce_ratio, in_channels=in_channels,
                                    out_channels=out_channels, kernel_per_layer=kernel_per_layer, 
                                    kernel_size=kernel_size_2, padding=padding_2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        y = self.channel_shuffle(x)
        y = self.pointwise(x)
        y_1 = self.s2block_1(y)
        y_2 = self.s2block_2(y)
        y = torch.cat((y_1, y_2), 1)
        y =  x + y
        return self.prelu(y)


################################################################
#                                                              #
#    - Define  Deepthwise convolution SE                       #
#                                                              #
# ##############################################################


"""
    - SquueezeBlock
"""
class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.tr = ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride)
        self.down = nn.Conv2d(in_channels=out_channels, out_channels= out_channels // 2, kernel_size=1, stride=1)
        self.up = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tr(x)
        shape = x.shape
        y = nn.AvgPool2d(kernel_size=shape[3])(x)
        y = self.down(y)
        y = self.up(y)
        y = self.sigmoid(y)
        return y * x


"""
    - DeepthwiseConvSEBlock
"""
class DeepthwiseConvSEBlock(nn.Module):

    def __init__(self, in_channels, kernel_per_layer, 
                out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.deepthwise = Deepthwise(in_channels=in_channels, kernel_per_layer=kernel_per_layer,
                                     kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=in_channels)
        self.prelu = nn.PReLU()
        self.pointwise = Pointwise(in_channels=in_channels, out_channels=in_channels)
        self.squeeze_block = SqueezeBlock(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        x = self.deepthwise(x)
        x = self.norm(x)
        x = self.prelu(x)
        x = self.pointwise(x)
        x = self.squeeze_block(x)
        return x



################################################################
#                                                              #
#    - SiNet                                                   #
#                                                              #
# ##############################################################

from .model import Model

class Encoder(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_1 = ConvBlock(in_channels=3, out_channels=12, kernel_size=3, stride=2)
        self.deepthwise_conv_se_block_1 = DeepthwiseConvSEBlock(in_channels = 12, out_channels = 16,
                                            kernel_per_layer=1 , kernel_size=3, stride=2)
        self.deepthwise_conv_se_block_2 = DeepthwiseConvSEBlock(in_channels = 16, out_channels = 48, 
                                            kernel_per_layer =1, kernel_size=3, stride=1)
        self.s2_module_1 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=48, kernel_per_layer=3, 
                                kernel_size_1=3, padding_1=1, kernel_size_2=5, padding_2=1)
        self.s2_module_2 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=48, kernel_per_layer=3,
                            kernel_size_1=3, padding_1=1, kernel_size_2=3, padding_2=1)
        self.deepthwise_conv_se_block_3 = DeepthwiseConvSEBlock(in_channels=64, out_channels=48, 
                                            kernel_per_layer=1, kernel_size=3, stride=2)
        self.deepthwise_conv_se_block_4 = DeepthwiseConvSEBlock(in_channels=48, out_channels=96, 
                                            kernel_per_layer=1, kernel_size=3, stride=1)
        self.s2_module_3 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=5, padding_2=1)
        self.s2_module_4 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=3, padding_2=1)
        self.s2_module_5 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=5, padding_2=1)
        self.s2_module_6 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=3, padding_2=1)
        self.s2_module_7 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=5, padding_2=1)
        self.s2_module_8 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=3, padding_2=1)
        self.s2_module_9 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=5, padding_2=1)
        self.s2_module_10 = S2Module(n_groups=4, reduce_ratio=2, in_channels_conv=96, kernel_per_layer=3, 
                                    kernel_size_1=3, padding_1=1, kernel_size_2=3, padding_2=1)
        # self.out = Pointwise(in_channels=144, out_channels=num_classes)
        self.out = DeepthwiseSeparableConv(in_channels=144, out_channels=num_classes, kernel_per_layer=1, 
                                    kernel_size=3, padding=0)
    
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.deepthwise_conv_se_block_1(x_1)
        x_3 = self.deepthwise_conv_se_block_2(x_2)
        x_4 = self.s2_module_1(x_3)
        x_5 = self.s2_module_2(x_4)
        x_concat_1 = torch.cat((x_2, x_5), 1)
        x_6 = self.deepthwise_conv_se_block_3(x_concat_1)
        x_7 = self.deepthwise_conv_se_block_4(x_6)
        x_8 = self.s2_module_3(x_7)
        x_9 = self.s2_module_4(x_8)
        x_10= self.s2_module_5(x_9)
        x_11 = self.s2_module_6(x_10)
        x_12 = self.s2_module_7(x_11)
        x_13 = self.s2_module_8(x_12)
        x_14 = self.s2_module_9(x_13)
        x_15 = self.s2_module_10(x_14)
        x_concat_2 = torch.cat((x_6,x_15), 1)
        x_17 = self.out(x_concat_2)
        return x_17, x_5

class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.up_sampling_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.norm_1 = nn.BatchNorm2d(num_features=num_classes)
        self.softmax = nn.Softmax(1)
        self.pointwise = Pointwise(in_channels=48, out_channels=num_classes)
        self.relu = nn.ReLU()
        self.up_sampling_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.norm_2 = nn.BatchNorm2d(num_features=num_classes)
        self.conv = ConvBlock(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1)
        self.up_sampling_3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.norm_3 = nn.BatchNorm2d(num_features=num_classes)

    def forward(self, x, x_5_):
        x_1 = self.up_sampling_1(x)
        x_2 = self.norm_1(x_1)
        x_3 = torch.max(self.softmax(x_2), dim=1)[0]
        blocking_map = (1 - x_3).unsqueeze(1).expand(x_2.shape)
        x_pointwise = self.pointwise(x_5_)
        x_4 = x_pointwise * blocking_map
        x_6 = x_4 + x_2 
        x_7 = self.relu(x_6)
        x_8 = self.up_sampling_2(x_7)
        x_9 = self.norm_2(x_8)
        x_10 = self.relu(x_9)
        x_11 = self.conv(x_10)
        x_12 = self.up_sampling_3(x_11)
        x_13 = self.norm_3(x_12)
        x_14 = self.relu(x_13)
        return self.softmax(x_14)
        
class SiNetModel(Model):

    def __init__(self, num_classes):
        super().__init__()
        self.Encoder = Encoder(num_classes)
        self.Decoder = Decoder(num_classes)

    def forward(self, x):
        features, x_5 = self.Encoder(x)
        return self.Decoder(features, x_5)


