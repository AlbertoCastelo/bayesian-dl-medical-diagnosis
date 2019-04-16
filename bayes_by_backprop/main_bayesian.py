import time

import math
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from BayesianModels.Bayesian4Conv3FC import BBB4Conv3FC
from BayesianModels.BayesianAlexNet import BBBAlexNet
from BayesianModels.BayesianLeNet import BBBLeNet
from bayes_by_backprop.configuration.loader import load_configuration
from dataset.histopathologic_cancer_dataset import HistoPathologicCancer
from dataset.x_ray_binary import XRayBinary
import torch
import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F


# model_type = 'densenet'
from utils.BBBlayers import GaussianVariationalInference

model_type = '3conv3fc'
is_resume = False

dataset = 'cancer'

is_debug = True

n_epochs = 100
lr = 0.1


config = load_configuration(filename=f'bbb-{model_type}-{dataset}.json')
batch_size = config['batch_size']
img_size = config['img_size']
n_channels = config['n_channels']
n_samples = config['n_samples']
beta_type = config['beta_type']


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
        model = BBBLeNet(num_classes, n_channels)
    elif model_type == 'alexnet':
        model = BBBAlexNet(num_classes, n_channels)
    elif model_type == '3conv3fc':
        model = BBB3Conv3FC(num_classes, n_channels)

    elif model_type == '4conv3fc':
        model = BBB4Conv3FC(num_classes, n_channels)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet /SqueezeNet/ 3Conv3FC')
    model.cuda()

else:
    # load from file
    pass

# define Variational Inference
vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())



# Define Training and Testing
optimizer = SGD([
    {'params': model.parameters()}
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


def validation(data_loader, dataset_type='Validation'):
    model.eval()
    test_loss = 0
    conf = []
    total = 0
    m = math.ceil(len(data_loader) / batch_size)
    correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):

        x = data.view(-1, n_channels, img_size, img_size).repeat(n_samples, 1, 1, 1).cuda()
        y = target.repeat(n_samples).cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
            outputs, kl = model.probforward(x)

            if beta_type is "Blundell":
                beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
            elif beta_type is "Soenderby":
                beta = min(epoch / (n_epochs // 4), 1)
            elif beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0

            loss = vi(outputs, y, kl, beta)
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            preds = F.softmax(outputs, dim=1)
            results = torch.topk(preds.cpu().data, k=1, dim=1)

            conf.append(results[0][0].item())
            total += target.size(0)
            correct += float(predicted.eq(y.data).cpu().sum())

    # p_hat = np.array(conf)
    # confidence_mean = np.mean(p_hat, axis=0)
    # confidence_var = np.var(p_hat, axis=0)
    # epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    # aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    accuracy = round(100. * correct / float(len(data_loader.dataset)) / n_samples, 3)
    print(f'{dataset_type} set: Accuracy: {correct/n_samples}/{len(data_loader.dataset)} ({accuracy}%)')
    return accuracy


def train(epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(train_set) / batch_size)

    print(f'Epoch: {epoch}')
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):

        x = data.view(-1, n_channels, img_size, img_size).repeat(n_samples, 1, 1, 1).cuda()
        y = target.repeat(n_samples).cuda()
        x, y = Variable(x), Variable(y)

        # choose beta
        if beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif beta_type is "Soenderby":
            beta = min(epoch / (n_epochs // 4), 1)
        elif beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        # Forward Propagation
        # x, y = Variable(x), Variable(y)
        outputs, kl = model.probforward(x)

        loss = vi(outputs, y, kl, beta)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += float(predicted.eq(y.data).cpu().sum())

        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))
    accuracy = round(100. * correct / float(len(train_loader.dataset)) / n_samples, 3)
    print(f'Training set: Accuracy: {correct/n_samples}/{len(train_loader.dataset)} ({accuracy}%)')

    return accuracy


# Train the Model
print('\n[Phase 3] : Training Model')
accuracy_val_top = 0
train_acc = []
validation_acc = []
time_epoch = []
for epoch in range(1, n_epochs + 1):
    start_time = time.time()
    scheduler.step()

    train_acc.append(train(epoch))
    # train_acc.append(validation(data_loader=train_loader, dataset_type='Training'))

    validation_acc_epoch = validation(data_loader=val_loader, dataset_type='Validation')
    validation_acc.append(validation_acc_epoch)

    if validation_acc_epoch > accuracy_val_top:
        print('Saving Model')
        state_dict = model.state_dict()
        torch.save({'model': state_dict}, f'bbb-{model_type}-{dataset}.dat')

    time_epoch.append(time.time() - start_time)

df_metric_training = pd.DataFrame({'epoch': list(range(1, len(train_acc) + 1)),
                                   'train_acc': train_acc,
                                   'validation_acc': validation_acc})
df_metric_training.to_csv(f'./bayes-{model_type}-{dataset}.csv', index=False)
