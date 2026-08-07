import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class MultiTask1DCNN(nn.Module):
    def __init__(self, num_type_classes=11, num_section_classes=12, dropout=0.2):
        super().__init__()
        self.layer1 = ResidualBlock1D(18, 64)
        self.pool1  = nn.MaxPool1d(2)
        self.drop1  = nn.Dropout(dropout)

        self.layer2 = ResidualBlock1D(64, 128)
        self.pool2  = nn.MaxPool1d(2)
        self.drop2  = nn.Dropout(dropout)

        self.layer3 = ResidualBlock1D(128, 256)
        self.layer4 = ResidualBlock1D(256, 256)
        self.pool4  = nn.MaxPool1d(2)
        self.drop4  = nn.Dropout(dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.type_head = nn.Linear(256, num_type_classes)
        self.section_head = nn.Linear(256, num_section_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(self.layer1(x)))
        x = self.drop2(self.pool2(self.layer2(x)))
        x = self.layer3(x)
        x = self.drop4(self.pool4(self.layer4(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.type_head(x), self.section_head(x)


if __name__ == "__main__":
    model = MultiTask1DCNN()
    dummy = torch.randn(4, 18, 333)
    type_out, section_out = model(dummy)
    print(type_out.shape, section_out.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
