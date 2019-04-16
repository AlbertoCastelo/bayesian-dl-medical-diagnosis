from gpytorch import settings
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from deep_gp.configuration.loader import load_configuration
from dataset.histopathologic_cancer_dataset import HistoPathologicCancer
from dataset.x_ray_binary import XRayBinary
import torch
import torchvision.transforms as transforms
import gpytorch


# parameters
from deep_gp.models.densenet import DenseNetFeatureExtractor
from deep_gp.models.deep_kernel_model import DKLModel
from deep_gp.models.resnet18 import ResNet18FeatureExtractorBernoulli
from deep_gp.models.resnet_bw import ResNetBWFeatureExtractor

# model_type = 'densenet'
model_type = 'resnet18'

# dataset = 'x_ray_binary'
dataset = 'cancer'

is_debug = True

n_epochs = 100
lr = 0.1


config = load_configuration(filename=f'bayes-bernouilli-{model_type}-{dataset}.json')
batch_size = config['batch_size']
img_size = config['img_size']
n_channels = config['n_channels']
num_likelihood_samples = config['num_likelihood_samples']

# n of samples used
settings.num_likelihood_samples._set_value(num_likelihood_samples)

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
    feature_extractor = ResNet18FeatureExtractorBernoulli(pretrained=True).cuda()
    num_features = feature_extractor.fc.out_features

# define model
model = DKLModel(feature_extractor, num_dim=num_features).cuda()
likelihood = gpytorch.likelihoods.BernoulliLikelihood().cuda()


# Define Training and Testing
optimizer = SGD([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


def validation(data_loader, dataset_type='Validation'):
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in data_loader:
        data, target = data.cuda(), target.float().cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.mean.ge(0.5).float()
            # pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    accuracy = 100. * correct / float(len(data_loader.dataset))
    print(f'{dataset_type} set: Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy}%)')
    return accuracy


def train(epoch):
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
    print(f'Epoch: {epoch}')
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.float().cuda()
        optimizer.zero_grad()
        output = model(data)
        output_2 = likelihood(model(data))
        pred = output.probs.float()
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
        validation(data_loader=train_loader, dataset_type='Training')
        accuracy_val = validation(data_loader=val_loader, dataset_type='Validation')
        if accuracy_val > accuracy_val_top:
            print('Saving Model')
            state_dict = model.state_dict()
            likelihood_state_dict = likelihood.state_dict()
            torch.save({'model': state_dict, 'likelihood': likelihood_state_dict},
                       f'bayes-bernoulli-{model_type}-{dataset}.dat')
