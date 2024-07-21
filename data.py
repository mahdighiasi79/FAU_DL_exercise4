from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import skimage

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = tv.io.read_image(self.data["filename"][index])
        crack = int(self.data["crack"][index])
        inactive = int(self.data["inactive"][index])
        image = skimage.color.gray2rgb(image)[0]
        image = self._transform(image)
        label = torch.tensor([crack, inactive])
        return image, label
