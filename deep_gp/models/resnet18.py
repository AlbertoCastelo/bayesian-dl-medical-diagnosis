from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet18FeatureExtractor(ResNet):
    '''
    Feature Extractor from ResNet 18
    '''
    def __init__(self, num_classes, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained
        super(ResNet18FeatureExtractor, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2],)

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet18FeatureExtractorBernoulli(ResNet):
    '''
    Feature Extractor from ResNet 18
    '''
    def __init__(self, num_classes=1, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained
        super(ResNet18FeatureExtractorBernoulli, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2],
                                                                num_classes=num_classes)

        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
            state_dict['fc.bias'] = torch.rand(num_classes)
            state_dict['fc.weight'] = torch.rand((num_classes, 512))
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
