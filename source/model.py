import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Implementation inspired by Aladdin Persson - PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby
class CNN_with_Unet(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=1,
        features=[64, 128, 256, 512],
        num_of_class=2,
        dist_from_center=10,
        drop_out=0.3,
        hidden_nodes=512,
    ):
        # UNET Convolution
        super(CNN_with_Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

            # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Fully Connected Layer
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=(2 * dist_from_center) ** 2, out_features=hidden_nodes
        )
        self.drop1 = nn.Dropout(p=drop_out)

        self.fc2 = nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes)
        self.drop2 = nn.Dropout(p=drop_out)

        self.out = nn.Linear(in_features=hidden_nodes, out_features=num_of_class)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.out(x)

        return x
