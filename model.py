import torch
import torch.nn as nn
import torchvision.models as models


# class ResNet50(nn.Module):

#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()
#         self.model = models.resnet50()
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc = nn.Linear(512 * 4, num_classes)

#     def forward(self, x):
#         return self.model(x)


class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        weight = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:, 0] = weight[:, 0]
        self.model.fc = nn.Linear(512, 1, bias=True)

    def forward(self, x):
        return self.model(x)


# class ResNetTest(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNetTest, self).__init__()
#         original_model = models.resnet18(pretrained=True)
#         conv_in = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.features = nn.Sequential(
#             conv_in,
#             *list(original_model.children())[1:-1]
#         )
#         # self.fc = nn.Linear(512, 4, bias=True)

#     def forward(self, x):
#         return self.features(x)
