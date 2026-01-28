import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # bias=False because batch norm does it (contains learnable B)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=16, num_classes=100):
        super(ResNet, self).__init__()
        # small number (16) because of small dataset
        self.in_channels = in_channels

        # self.in channels used for more flexibility in architecture's width
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stages = nn.ModuleList()

        out_channels = self.in_channels
        # self._make_layer modifies self.in_channels; loop makes any number of layers possible
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            if i != 0:
                out_channels *= 2

            self.stages.append(
                self._make_layer(block, out_channels, num_blocks, stride)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

        # for appropriate layers initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # initializes weights with mean=0 and variance depending on number of output size
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # initialize to "identity" at the beginning, makes early learning more stable
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # in case size of residual connection value passed needs to be modified
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet34(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet56(num_classes=100):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

