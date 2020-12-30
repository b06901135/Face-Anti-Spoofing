import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import random
import numpy as np
from PIL import Image


class OuluDataset(Dataset):
    def __init__(self, image_dir, argument=True, image_dim=512, limit_num=None, return_label=True):
        self.argument = argument
        self.image_dim = image_dim
        self.return_label = return_label

        self.sub_dirs = sorted(os.listdir(image_dir))
        self.files = []
        self.labels = []
        for sub_dir in self.sub_dirs:
            temp = sorted(os.listdir(os.path.join(image_dir, sub_dir)))
            temp = [os.path.join(image_dir, sub_dir, file) for file in temp]
            self.files.append(temp[:limit_num] if limit_num is not None else temp)
            if self.return_label:
                self.labels.append(int(sub_dir.split('_')[-1]) - 1)

    def transform(self, files):
        images = []
        for file in files:
            images.append(Image.open(file))

        L = len(images)
        # Resize
        for i in range(L):
            images[i] = images[i].resize((self.image_dim, self.image_dim))

        if self.argument:
            # Horizontal flip
            if random.random() > 0.5:
                for i in range(L):
                    images[i] = transforms.functional.hflip(images[i])

            # Random crop
            # s = 512
            # c = self.crop_size
            # i = torch.randint(0, s - c + 1, size=(1, )).item()
            # j = torch.randint(0, s - c + 1, size=(1, )).item()

            # x = G.crop(x, i, j, c, c)
            # y = G.crop(y, i, j, c, c)

            # Color jitter
            alpha = 0.2
            for i in range(L):
                images[i] = transforms.functional.adjust_brightness(images[i], random.uniform(1 - alpha, 1 + alpha))
                images[i] = transforms.functional.adjust_contrast  (images[i], random.uniform(1 - alpha, 1 + alpha))
                images[i] = transforms.functional.adjust_saturation(images[i], random.uniform(1 - alpha, 1 + alpha))

        # To tensor
        for i in range(L):
            images[i] = transforms.functional.to_tensor(images[i])

        # Normalize
        for i in range(L):
            images[i] = transforms.functional.normalize(images[i], mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])

        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3) # change channel and frame order
        return images

    def __getitem__(self, idx):
        x = self.transform(self.files[idx])
        return (x, self.labels[idx]) if self.return_label else x

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = OuluDataset('oulu_npu_cropped/train', limit_num=10)
    dataloader = DataLoader(dataset, batch_size=8)
    (x, y) = next(iter(dataloader))
    print(x.size(), y)
