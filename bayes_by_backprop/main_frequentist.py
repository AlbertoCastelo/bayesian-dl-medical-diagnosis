import time

import math

from torch import nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from FrequentistModels import F3Conv3FC, AlexNet, LeNet
from FrequentistModels.F4Conv3FC import F4Conv3FC
from bayes_by_backprop.configuration.loader import load_configuration
from dataset.histopathologic_cancer_dataset import HistoPathologicCancer
from dataset.x_ray_binary import XRayBinary
import torch
import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F


model_type = '3conv3fc'
is_resume = False

dataset = 'cancer'

is_debug = True

n_epochs = 10
lr = 0.001


config = load_configuration(filename=f'frequentist-{model_type}-{dataset}.json')
batch_size = config['batch_size']
img_size = config['img_size']
n_channels = config['n_channels']
weight_decay = config['weight_decay']

if is_debug:
    batch_size = 64

# aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [
    transforms.CenterCrop((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6976, 0.5404, 0.6897],
                         std=[0.2433, 0.2839, 0.2199])
]

transform_train = transforms.Compose(common_trans)
transform_val = transforms.Compose(common_trans)

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)

# Generate DataLoaders
print('\n[Phase 1]: Creating DataLoaders')

if dataset == 'x_ray_binary':
    data_path = '/home/alberto/Desktop/repos/random/xray-bayesian-dl/data/x_ray_data'
    # data_path = '/home/alberto/Desktop/repos/bayesian-deep-learning/bayesian-dl-xray/data/x_ray_data'
    train_set = XRayBinary(path=data_path, img_size=img_size, is_train=True, transform=transform_train, is_debug=is_debug)
    val_set = XRayBinary(path=data_path, img_size=img_size, is_train=False, transform=transform_val, is_debug=is_debug)
    num_classes = 2

elif dataset == 'cancer':
    data_path = '/home/alberto/Desktop/datasets/histopathologic-cancer-detection/'
    train_set = HistoPathologicCancer(path=data_path, img_size=img_size, dataset_type='train',
                                      transform=transform_train,
                                      is_debug=is_debug)
    val_set = HistoPathologicCancer(path=data_path, img_size=img_size, dataset_type='validation',
                                    transform=transform_val,
                                    is_debug=is_debug)
    num_classes = 2

else:
    raise Exception

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

print('\n[Phase 2] : Model setup')

# Return network & file name
if not is_resume:
    if model_type == 'lenet':
        model = LeNet(num_classes, n_channels)
    elif model_type == 'alexnet':
        model = AlexNet(num_classes, n_channels)
    elif model_type == '3conv3fc':
        model = F3Conv3FC(num_classes, n_channels)

    elif model_type == '4conv3fc':
        model = F4Conv3FC(num_classes, n_channels)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet /SqueezeNet/ 3Conv3FC')
    model.cuda()

else:
    # load from file
    pass

# define Loss Criteria
criteria = nn.CrossEntropyLoss()


# Define Training and Testing
# optimizer = SGD([
#     {'params': model.parameters()}
# ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
# scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

def validation(data_loader, dataset_type='Validation'):
    model.eval()
    validation_loss = 0
    conf = []
    total = 0
    m = math.ceil(len(data_loader) / batch_size)
    correct = 0

    for batch_idx, (x, y) in enumerate(data_loader):

        with torch.no_grad():
            x, y = Variable(x.cuda()), Variable(y.cuda())
            outputs = model(x)
            loss = criteria(outputs, y)

            validation_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            preds = F.softmax(outputs, dim=1)
            results = torch.topk(preds.cpu().data, k=1, dim=1)

            conf.append(results[0][0].item())
            total += y.size(0)
            correct += float(predicted.eq(y.data).cpu().sum())

    # p_hat = np.array(conf)
    # confidence_mean = np.mean(p_hat, axis=0)
    # confidence_var = np.var(p_hat, axis=0)
    # epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    # aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    accuracy = round(100. * correct / float(len(data_loader.dataset)), 3)
    print(f'{dataset_type} set: Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy}%)')
    return accuracy


def train(epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(train_set) / batch_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f'Epoch: {epoch}')
    train_loss = 0.
    for batch_idx, (x, y) in enumerate(train_loader):

        x, y = Variable(x.cuda()), Variable(y.cuda())

        optimizer.zero_grad()

        outputs = model(x)  # Forward Propagation
        loss = criteria(outputs, y)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += float(predicted.eq(y.data).cpu().sum())

        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))
    accuracy = round(100. * correct / float(len(train_loader.dataset)), 3)
    print(f'Training set: Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy}%)')

    return accuracy


# Train the Model
print('\n[Phase 3] : Training Model')
accuracy_val_top = 0
train_acc = []
validation_acc = []
time_epoch = []
for epoch in range(1, n_epochs + 1):
    start_time = time.time()
    # scheduler.step()

    train_acc.append(train(epoch))
    # train_acc.append(validation(data_loader=train_loader, dataset_type='Training'))

    validation_acc_epoch = validation(data_loader=val_loader, dataset_type='Validation')
    validation_acc.append(validation_acc_epoch)

    if validation_acc_epoch > accuracy_val_top:
        print('Saving Model')
        state_dict = model.state_dict()
        torch.save({'model': state_dict}, f'frequentist-{model_type}-{dataset}.dat')

    epoch_time = time.time() - start_time
    time_epoch.append(epoch_time)
    print(f'Time at epoch: {epoch_time}')

    df_metric_training = pd.DataFrame({'epoch': list(range(1, len(train_acc) + 1)),
                                       'train_acc': train_acc,
                                       'validation_acc': validation_acc,
                                       'time': time_epoch})
    df_metric_training.to_csv(f'./frequentist-{model_type}-{dataset}.csv', index=False)
