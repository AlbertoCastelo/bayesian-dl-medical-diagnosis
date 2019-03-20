import os
from glob import glob

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class HistoPathologicCancer(Dataset):
    """Histo-Pathologic Cancer Dataset with binary labeling: Finding/No-Finding"""

    def __init__(self, path=None, img_size=64, is_train=False, transform=None, n_channels=1, is_debug=False):
        self.path = path if path is not None else '/home/alberto/Desktop/datasets/histopathologic-cancer-detection/'
        self.img_size = img_size
        self.is_train = is_train
        self.n_channels = n_channels
        self.is_debug = is_debug

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.ToTensor()])

        self.transform_target = transforms.ToTensor()

        self.df_label_img = pd.read_csv(os.path.join(self.path, 'train_labels.csv'))

        if is_debug:
            self.df_label_img = self.df_label_img[:512]

    def __len__(self):
        return len(self.df_label_img)

    def __getitem__(self, idx):
        img = self.get_rgb_image_from_file(os.path.join(self.path, 'train', ''.join([self.df_label_img[idx], '.tif'])))
        img = self.transform(img)

        label = np.array(self.df_label_img[idx])
        label = torch.from_numpy(label)
        return img, label

    def load_image_from_file(self, filename):
        return Image.open(self.image_paths[filename])

    def get_rgb_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('RGB')

    def get_grayscale_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('L')
