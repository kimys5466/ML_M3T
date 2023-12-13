import torch
from torch import nn
from torch import Tensor
from typing import Callable, List, Optional, Type, Union

def conv3x3(in_planes : int, out_planes : int, stride : int=1, groups : int =1, dilation : int=1) -> nn.Conv2d:
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation =dilation)

def conv1x1(in_planes : int, out_planes : int, stride : int=1) -> nn.Conv2d:
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            img_size: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        # Normalization layer
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer([planes,img_size,img_size])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer([planes,img_size,img_size])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # downsampling이 필요한 경우 downsample 레이어를 block에 인자로 넣어주어야 함
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual connection
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        img_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer([width, img_size, img_size])
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer([width, img_size, img_size])
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer([planes * self.expansion, img_size, img_size])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm([64,64,64])
            residual_norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.LayerNorm  # batch norm layer
        self.inplanes = 64  # input shape
        self.dilation = 1
        self.groups = 1

        # input block
        self.conv1 = nn.Conv2d(128, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # input channel에 따라 바뀌는 부분
        self.bn1 = norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual blocks
        self.layer1 = self._make_layer(block, 32, 32, layers[0])
        self.layer2 = self._make_layer(block, 32, 32, layers[1], stride=1, dilate=False)
        self.layer3 = self._make_layer(block, 64, 32, layers[2], stride=1, dilate=False)
        self.layer4 = self._make_layer(block, 128, 32, layers[3], stride=1, dilate=False)
        self.conv2 = nn.Conv2d(128, self.inplanes, kernel_size=32, stride=1, padding=0, bias=False) # Input image size에 따라 kernel size 변경 가능
        # self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=2, stride=2, padding=0, bias=False)

        # weight initalizaiton
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # zero-initialize the last BN in each residual branch
            # so that the residual branch starts with zero, and each residual block behaves like an identity
            # Ths improves the model by 0.2~0.3%
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, img_size : int, blocks: int, stride: int = 1, dilate: bool = False,) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        # downsampling 필요한 경우 downsample layer 생성
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer([planes, img_size, img_size]),
            )
        layers = []
        layers.append(block(self.inplanes, planes, img_size, stride, downsample, self.groups, self.dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, img_size, groups=self.groups, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        #print('input shape:', x.shape)
        x = self.conv1(x)
        #print('conv1 shape:', x.shape)
        x = self.bn1(x)
        #print('bn1 shape:', x.shape)
        x = self.relu(x)
        #print('relu shape:', x.shape)
        x = self.maxpool(x)
        #print('maxpool shape:', x.shape)

        x = self.layer1(x)
        #print('layer1 shape:', x.shape)
        x = self.layer2(x)
        #print('layer2 shape:', x.shape)
        x = self.layer3(x)
        #print('layer3 shape:', x.shape)
        x = self.layer4(x)
        # print('layer4 shape:', x.shape)
        y = self.conv2(x)
        # print('Attention layer:', y.shape)
        z = x*y
        # print('output:', z.shape)
        return z
        # return torch.reshape(self.conv3(z),[x.shape[0],128,256])

# model = ResNet18(BasicBlock, [2,2,2,2])
# x = torch.randn(100,128,128,128)
# x = model(x)
# print(x.shape)