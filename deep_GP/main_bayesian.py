from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from deep_GP.configuration.loader import load_configuration
from deep_GP.dataset.histopathologic_cancer_dataset import HistoPathologicCancer
from deep_GP.dataset.x_ray_binary import XRayBinary
import torch
import torchvision.transforms as transforms
import gpytorch


# parameters
from deep_GP.models.densenet import DenseNetFeatureExtractor
from deep_GP.models.deep_kernel_model import DKLModel
from deep_GP.models.resnet18 import ResNet18FeatureExtractor
from deep_GP.models.resnet_bw import ResNetBWFeatureExtractor

# model_type = 'densenet'
model_type = 'resnet18'

# dataset = 'x_ray_binary'
dataset = 'cancer'

is_debug = False

n_epochs = 100
lr = 0.1


config = load_configuration(filename=f'bayes-{model_type}-{dataset}.json')
batch_size = config['batch_size']
img_size = config['img_size']
n_channels = config['n_channels']

if is_debug:
    batch_size = 64

# aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6], std=[0.2])
]

transform_train = transforms.Compose(common_trans)
transform_val = transforms.Compose(common_trans)

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)

# Generate DataLoaders
print('Creating DataLoaders')
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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

# create Feature Extractor
if model_type == 'densenet':
    feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), n_channels=1, num_classes=num_classes).cuda()
    num_features = feature_extractor.classifier.in_features
elif model_type == 'resnet':
    feature_extractor = ResNetBWFeatureExtractor(num_classes=2).cuda()
    num_features = feature_extractor.fc.in_features

elif model_type == 'resnet18':
    feature_extractor = ResNet18FeatureExtractor(num_classes=2, pretrained=True).cuda()
    num_features = feature_extractor.fc.in_features

# define model
model = DKLModel(feature_extractor, num_dim=num_features).cuda()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, n_classes=num_classes).cuda()


# Define Training and Testing
optimizer = SGD([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


def validation():
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    accuracy = 100. * correct / float(len(val_loader.dataset))
    print(f'Test set: Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy}%)')
    return accuracy

def train(epoch):
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
    print(f'Epoch: {epoch}')
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))


# Train the Model
print('Training Model')
accuracy_val_top = 0
for epoch in range(1, n_epochs + 1):
    scheduler.step()

    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
        accuracy_val = validation()
        if accuracy_val > accuracy_val_top:
            print('Saving Model')
            state_dict = model.state_dict()
            likelihood_state_dict = likelihood.state_dict()
            torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, f'bayes-{model_type}-{dataset}.dat')
