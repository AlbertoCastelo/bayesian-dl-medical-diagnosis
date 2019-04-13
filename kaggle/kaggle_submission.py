from deep_gp.configuration.loader import load_configuration
from deep_gp.dataset.histopathologic_cancer_dataset import HistoPathologicCancer
from gpytorch import settings
import torch
import torchvision.transforms as transforms
import gpytorch
import pandas as pd

# parameters
from deep_gp.models.densenet import DenseNetFeatureExtractor
from deep_gp.models.deep_kernel_model import DKLModel
from deep_gp.models.resnet18 import ResNet18FeatureExtractor
from deep_gp.models.resnet_bw import ResNetBWFeatureExtractor

# model_type = 'densenet'
model_type = 'resnet18'

# dataset = 'x_ray_binary'
dataset = 'cancer'

is_debug = False


config = load_configuration(filename=f'bayes-{model_type}-{dataset}.json', path='./../deep_gp/configuration')
batch_size = 896
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

if dataset == 'cancer':
    data_path = '/home/alberto/Desktop/datasets/histopathologic-cancer-detection/'
    test_set = HistoPathologicCancer(path=data_path, img_size=img_size, dataset_type='test', transform=transform_val,
                                     is_debug=is_debug)
    num_classes = 2

else:
    raise Exception

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

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


# load model
model_filename = f'./../deep_gp/bayes-{model_type}-{dataset}-100_samples.dat'
print(f'Loading model from {model_filename}')
trained_model = torch.load(model_filename)
model.load_state_dict(trained_model['model'])
likelihood.load_state_dict(trained_model['likelihood'])

# set to eval mode
model.eval()
likelihood.eval()


def test_generation():
    model.eval()
    likelihood.eval()

    correct = 0
    chunks = []
    for data, ids in test_loader:
        data = data.cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            result_chunk = pd.DataFrame({'id': ids,
                                         'label': pred.cpu()})

            chunks.append(result_chunk)

    return pd.concat(chunks, sort=False)


results = test_generation()
results.to_csv('./submission.csv', index=False)
