import os
from glob import glob

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image

FILENAME_LABELS_BINARY = 'Data_Entry_2017_Binary.csv'


class XRayBinary(Dataset):
    """X-Ray Dataset with binary labeling: Finding/No-Finding"""

    def __init__(self, path, img_size=64, is_train=False, transform=None, n_channels=1, is_debug=False):
        self.path = path
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

        self.x_ray_df = pd.read_csv(os.path.join(self.path, FILENAME_LABELS_BINARY))
        self.x_ray_df = self.x_ray_df.loc[self.x_ray_df['Train'] == is_train]

        self.image_paths = {os.path.basename(x): x for x in
                            glob(os.path.join(self.path, 'images', '*.png'))}

        filenames_selected = set(self.x_ray_df['Image Index'].values)
        filenames_all = set(self.image_paths.keys())
        filenames_to_remove = list(filenames_all-filenames_selected)
        for filename in filenames_to_remove:
            del self.image_paths[filename]
        assert(set(self.image_paths.keys()) == filenames_selected)

        self.x_ray_df = self.x_ray_df.set_index('Image Index')
        self.x_ray_df['target'] = 0
        self.x_ray_df.loc[self.x_ray_df['Binary Labels'] == 'Finding', ['target']] = 1
        self.img_filenames = list(self.image_paths.keys())
        # TODO: make lazy loader
        if is_debug:
            self.img_filenames = self.img_filenames[:512]
        self.img_label = np.array(
            [self.x_ray_df.loc[filename, ['target']].values[0] for filename in self.img_filenames])

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        if self.n_channels == 1:
            img = self.get_grayscale_image_from_file(self.img_filenames[idx])
        elif self.n_channels == 3:
            img = self.get_rgb_image_from_file(self.img_filenames[idx])
        else:
            raise Exception

        img = self.transform(img)
        # if img.shape[0] > 1:
        #     print(self.img_filenames[idx])
        #     print(img.shape)
        label = np.array(self.img_label[idx])
        label = torch.from_numpy(label)
        return img, label

    def load_image_from_file(self, filename):
        return Image.open(self.image_paths[filename])

    def get_rgb_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('RGB')

    def get_grayscale_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('L')
