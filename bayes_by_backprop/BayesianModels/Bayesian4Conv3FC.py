import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer

class BBB4Conv3FC(nn.Module):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBB4Conv3FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 32, kernel_size=5, stride=1, padding=0)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = BBBConv2d(128, 256, kernel_size=5, stride=1, padding=0)
        self.soft4 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.conv5 = BBBConv2d(256, 512, kernel_size=5, stride=1, padding=0)
        # self.soft5 = nn.Softplus()
        # self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(3 * 3 * 256)
        self.fc1 = BBBLinearFactorial(3 * 3 * 256, 1000)
        self.soft6 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft7 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.conv3, self.soft3, self.pool3, self.conv4, self.soft4, self.pool4,
                  self.flatten, self.fc1, self.soft6,
                  self.fc2, self.soft7, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            # print(layer)
            # print(x.shape)
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        # print('logits', logits)
        return logits, kl
