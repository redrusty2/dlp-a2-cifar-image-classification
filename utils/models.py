from torch import nn
import torch


def init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input):
        out = self.features(input)
        out = self.adapt(out)
        out = self.classifier(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rl2 = nn.ReLU()

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.rl1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.residual(x)
        out = self.rl2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.features = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):
        out = self.init(input)
        out = self.features(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out


class ResNet18Dropout(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Dropout, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.features = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):
        out = self.init(input)
        out = self.features(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out


class ResNet18DropoutNoise(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18DropoutNoise, self).__init__()

        self.noise = GaussianNoise(0.1)

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.features = nn.Sequential(
            ResidualBlock(64, 64),
            GaussianNoise(0.1),
            ResidualBlock(64, 64),
            GaussianNoise(0.1),
            ResidualBlock(64, 128, stride=2),
            GaussianNoise(0.1),
            ResidualBlock(128, 128),
            GaussianNoise(0.1),
            ResidualBlock(128, 256, stride=2),
            GaussianNoise(0.1),
            ResidualBlock(256, 256),
            GaussianNoise(0.1),
            ResidualBlock(256, 512, stride=2),
            GaussianNoise(0.1),
            ResidualBlock(512, 512),
            GaussianNoise(0.1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):
        out = self.noise(input)
        out = self.init(out)
        out = self.features(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out


# From https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/2
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float32).to("cuda")

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        intermed_channels: int,
        stride: int = 1,
    ):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            intermed_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermed_channels)
        self.rl1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            intermed_channels,
            intermed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermed_channels)
        self.rl2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            intermed_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.rl3 = nn.ReLU()

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.rl1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rl2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.residual(x)
        out = self.rl3(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            BottleNeckBlock(64, 256, 64),
            BottleNeckBlock(256, 256, 64),
            BottleNeckBlock(256, 256, 64),
        )
        self.conv3 = nn.Sequential(
            BottleNeckBlock(256, 512, 128, stride=2),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
        )
        self.conv4 = nn.Sequential(
            BottleNeckBlock(512, 1024, 256, stride=2),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
        )
        self.conv5 = nn.Sequential(
            BottleNeckBlock(1024, 2048, 512, stride=2),
            BottleNeckBlock(2048, 2048, 512),
            BottleNeckBlock(2048, 2048, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, input):
        out = self.init(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out


class ResNet50Mod(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Mod, self).__init__()

        self.conv2 = nn.Sequential(
            BottleNeckBlock(3, 256, 64),
            BottleNeckBlock(256, 256, 64),
            BottleNeckBlock(256, 256, 64),
        )
        self.conv3 = nn.Sequential(
            BottleNeckBlock(256, 512, 128, stride=2),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
            BottleNeckBlock(512, 512, 128),
        )
        self.conv4 = nn.Sequential(
            BottleNeckBlock(512, 1024, 256, stride=2),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
            BottleNeckBlock(1024, 1024, 256),
        )
        self.conv5 = nn.Sequential(
            BottleNeckBlock(1024, 2048, 512, stride=2),
            BottleNeckBlock(2048, 2048, 512),
            BottleNeckBlock(2048, 2048, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, input):
        out = self.conv2(input)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out
