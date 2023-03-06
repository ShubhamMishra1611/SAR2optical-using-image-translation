import torch
import torch.nn as nn

class CNN_block(nn.Module):
    def __init__(self, input_channel, output_channel, stride = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=stride, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_channel = 3, features = [64, 128, 256, 512]) -> None:
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channel*2,
            out_channels=features[0],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2),
        )

        layers = [CNN_block(features[i-1], features[i], stride=1 if i == len(features)-1 else 2) for i in range(1,len(features))]
        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode = "reflect")
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1) # along the channel
        x = self.initial_block(x)
        return self.model(x)
    
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    predictions = model(x,y)
    print(predictions.shape)


if __name__ == "__main__":
    test()