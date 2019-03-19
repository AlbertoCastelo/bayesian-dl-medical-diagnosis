from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet18

from deep_GP.dataset.x_ray_binary import XRayBinary
import torch
import torchvision.transforms as transforms

from deep_GP.models.resnet34 import ResNet

# feature_extractor_type = 'densenet'
feature_extractor_type = 'resnet18'
is_debug = False

n_epochs = 100
lr = 0.1

if feature_extractor_type == 'resnet':
    batch_size = 256-16
    img_size = 224
    n_channels = 1
elif feature_extractor_type == 'resnet18':
    batch_size = 128
    img_size = 224
    n_channels = 3
else:
    raise Exception

if is_debug:
    batch_size = 64

# aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6]*n_channels, std=[0.2]*n_channels)
]
transform_train = transforms.Compose(common_trans)
transform_val = transforms.Compose(common_trans)

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)

# Generate DataLoaders
print('Creating DataLoaders')
# data_path = '/home/alberto/Desktop/repos/random/xray-bayesian-dl/data/x_ray_data'
data_path = '/home/alberto/Desktop/repos/bayesian-deep-learning/bayesian-dl-xray/data/x_ray_data'
train_set = XRayBinary(path=data_path, img_size=img_size, is_train=True, transform=transform_train,
                       n_channels=n_channels, is_debug=is_debug)
val_set = XRayBinary(path=data_path, img_size=img_size, is_train=False, transform=transform_val,
                     n_channels=n_channels, is_debug=is_debug)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
num_classes = 2

print('Creating Model')
# create Feature Extractor
if feature_extractor_type == 'resnet':
    model = ResNet(num_classes=num_classes).cuda()
if feature_extractor_type == 'resnet18':
    model = resnet18(pretrained=True).cuda()

# Define Training and Testing
optimizer = SGD([
    {'params': model.parameters()}
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
criterion = CrossEntropyLoss()


def validation():

    correct = 0
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(val_loader.dataset), 100. * correct / float(len(val_loader.dataset))
    ))


def train(epoch):

    print(f'Epoch: {epoch}')
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))


# Train the Model
print('Training Model')
for epoch in range(1, n_epochs + 1):
    scheduler.step()
    train(epoch)
    validation()

    state_dict = model.state_dict()
    torch.save({'model': state_dict}, 'x_ray_binary_checkpoint.dat')
