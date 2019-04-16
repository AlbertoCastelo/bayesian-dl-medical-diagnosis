import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class HistoPathologicCancer(Dataset):
    """Histo-Pathologic Cancer Dataset with binary labeling: Finding/No-Finding"""

    def __init__(self, path=None, img_size=64, dataset_type='train', transform=None, n_channels=3, is_debug=False):
        self.path = path if path is not None \
            else '/home/alberto/Desktop/datasets/histopathologic-cancer-detection/'
        self.img_size = img_size
        self.dataset_type = dataset_type
        if self.dataset_type not in ['train', 'validation', 'test']:
            raise RuntimeError

        if self.dataset_type in ['train', 'validation']:
            self.is_training = True
        else:
            self.is_training = False

        self.n_channels = n_channels
        self.is_debug = is_debug

        # define transformation
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.ToTensor()])

        self.transform_target = transforms.ToTensor()

        self.df_data = pd.read_csv(os.path.join(self.path, 'train_labels.csv'))

        if is_debug:
            self.df_data = self.df_data[:512]

        # create training/validation set
        train, validation_and_test = train_test_split(self.df_data, random_state=42, shuffle=True, test_size=0.4)
        validation, test = train_test_split(validation_and_test, random_state=42, shuffle=True, test_size=0.5)

        if self.dataset_type == 'train':
            self.df_data = train
        elif self.dataset_type == 'validation':
            self.df_data = validation
        elif self.dataset_type == 'test':
            self.df_data = test
        else:
            raise ValueError

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        item = self.df_data.iloc[idx]
        img = self.get_rgb_image_from_file(os.path.join(self.path, 'train', ''.join([item['id'],
                                                                                     '.tif'])))
        img = self.transform(img)
        label = np.array(item['label'])
        label = torch.from_numpy(label)
        return img, label

    def load_image_from_file(self, filename):
        return Image.open(filename)

    def get_rgb_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('RGB')

    def get_grayscale_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('L')
