from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random

import tarfile
import io
import os
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CommonVoiceUkr(Dataset):
    def __init__(self,
                 txt_path='filelist.txt',
                 img_dir='data',
                 pad_size=1024,
                 transform=None,
                 batch_size=1):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """
        df = pd.read_csv(txt_path, index_col=0)
        n = len(df)
        df = df[:(n - n%batch_size)]
        self.speech_text = df["sentence"]
        self.img_names = df["spectro_path"]
        self.pad_size = pad_size
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.get_image_selector = True if img_dir.__contains__('tar') else False
        self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None

    def get_image_from_tar(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = self.tf.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = Image.open(os.path.join(self.img_dir, name)).convert("L")
        return image

    def pad_spectro_length(self, X):
        T = X.shape[-1]
        half_pad = (self.pad_size - T) // 2
        left, right = (half_pad, half_pad) if T % 2 == 0 else (half_pad, half_pad + 1)
        X = F.pad(X, (left, right))
        return X

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        if index == (self.__len__() - 1) and self.get_image_selector:  # close tarfile opened in __init__
            self.tf.close()

        if self.get_image_selector:  # note: we prefer to extract then process!
            X = self.get_image_from_tar(self.img_names[index])
        else:
            X = self.get_image_from_folder(self.img_names[index])

        # Get you label here using available pandas functions
        Y = self.speech_text[index]

        if self.transform is not None:
            X = self.transform(X)
        else:
            X = self.to_tensor(X)

        X = self.pad_spectro_length(X)

        return X, Y

