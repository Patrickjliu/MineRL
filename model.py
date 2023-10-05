import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.lstm = nn.LSTM(input_size=512 * 4, hidden_size=512*4, num_layers=2, batch_first=True)
        self.fc1x_float = nn.Linear(512*4, 32)
        self.relu3x_float = nn.ReLU()
        self.fc2x_mean = nn.Linear(32, 1)
        self.fc2x_std = nn.Linear(32, 1)

        self.fc1y_float = nn.Linear(512*4, 32)
        self.relu3y_float = nn.ReLU()
        self.fc2y_mean = nn.Linear(32, 1)
        self.fc2y_std = nn.Linear(32, 1)

        self.fc1 = nn.Linear(512*4, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

        self.fcv = nn.Linear(512*4, 256)
        self.relu4 = nn.ReLU()
        self.fc_v = nn.Linear(256, 512)
        self.relu5 = nn.ReLU()
        self.fc_v_ap = nn.Linear(512, 512)
        self.relu6 = nn.ReLU()
        self.fc_v_a = nn.Linear(512, 1)
        self.fc_v_cx = nn.Linear(256, 1)
        self.fc_v_cy = nn.Linear(256, 1)

    def a(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x, _ = self.lstm(x)
        
        x_floatx = self.fc1x_float(x)
        x_floatx = self.relu3x_float(x_floatx)
        mean_x = torch.tanh(self.fc2x_mean(x_floatx))
        std_x = torch.sigmoid(self.fc2x_std(x_floatx)) + 1e-10  # Ensure std is positive

        x_floaty = self.fc1y_float(x)
        x_floaty = self.relu3y_float(x_floaty)
        mean_y = torch.tanh(self.fc2y_mean(x_floaty))
        std_y = torch.sigmoid(self.fc2y_std(x_floaty)) + 1e-10  # Ensure std is positive

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        logits = self.fc3(x)
        action_probs = torch.softmax(logits, dim=-1)

        return action_probs, mean_x, std_x, mean_y, std_y

    def v(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x, _ = self.lstm(x)

        x = self.fcv(x)
        x = self.relu4(x)
        v_a = self.fc_v(x)
        v_a = self.relu5(v_a)
        v_a = self.fc_v_ap(v_a)
        v_a = self.relu6(v_a)
        v_cx = self.fc_v_cx(x)
        v_cy = self.fc_v_cy(x)
        v_a = self.fc_v_a(v_a)
        return v_cx, v_cy, v_a

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

