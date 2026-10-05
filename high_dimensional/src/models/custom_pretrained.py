import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torchvision
import torch.nn.functional as F


class ResNet18PT(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18PT, self).__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)

        # Get number of input features for classifier
        num_ftrs = self.base_model.fc.in_features
        self.base_model = torch.nn.Sequential(*list(self.base_model.children())[:-1])

        self.classifier = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x, return_feat=False):
        features = self.base_model(x)
        out = self.classifier(torch.flatten(features, start_dim=1))
        if return_feat:
            return out, features.squeeze()
        else:
            return out


class VGG19BNPT(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(VGG19BNPT, self).__init__()

        self.base_model = torchvision.models.vgg19_bn(pretrained=True)

        # Get number of input features for classifier
        num_ftrs = self.base_model.classifier[-1].in_features
        self.base_model.classifier = torch.nn.Sequential(
            *list(self.base_model.classifier.children())[:-1]
        )
        self.classifier = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x, return_feat=False):
        features = self.base_model(x)
        out = self.classifier(features)
        if return_feat:
            return out, features.squeeze()
        else:
            return out
