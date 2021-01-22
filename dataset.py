import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import random
import numpy as np
from PIL import Image
import cv2


class VideoDataset(Dataset):
    def __init__(self, image_dir, argument=True, image_dim=112, limit_num=None, five_crop=None, return_label=True):
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

    def transform(self, files, random_crop=True):
        images = []
        for file in files:
            images.append(Image.open(file))

        L = len(images)
        # Resize
        resize_dim = int(self.image_dim * random.uniform(1.0, 1.5))
        for i in range(L):
            if random_crop and self.argument:
                images[i] = images[i].resize((resize_dim, resize_dim))
            else:
                images[i] = images[i].resize((self.image_dim, self.image_dim))

        if self.argument:
            # Horizontal flip
            if random.random() > 0.5:
                for i in range(L):
                    images[i] = transforms.functional.hflip(images[i])

            # Random crop
            if random_crop:
                s = resize_dim
                c = self.image_dim
                i = torch.randint(0, s - c + 1, size=(1, )).item()
                j = torch.randint(0, s - c + 1, size=(1, )).item()

                for i in range(L):
                    images[i] = transforms.functional.crop(images[i], i, j, c, c)

            # Color jitter
            # alpha = 0.2
            alpha = 0.4
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


class TextureDataset(Dataset):
    def __init__(self, image_dir, argument=True, image_dim=224, limit_num=None, five_crop=False, return_label=True):
        self.argument = argument
        self.image_dim = image_dim
        self.return_label = return_label

        self.sub_dirs = sorted(os.listdir(image_dir))
        self.files = []
        self.labels = []
        for sub_dir in self.sub_dirs:
            temp = sorted(os.listdir(os.path.join(image_dir, sub_dir)))
            temp = [os.path.join(image_dir, sub_dir, file) for file in temp]
            temp = temp[:limit_num] if limit_num is not None else temp
            self.files.extend(temp)
            if self.return_label:
                self.labels.extend([int(sub_dir.split('_')[-1]) - 1 for _ in range(len(temp))])

        if argument:
            self.transform = transforms.Compose([
                lambda file: Image.open(file),
                transforms.RandomCrop((self.image_dim, self.image_dim)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing()
            ])
        elif five_crop:
            self.transform = transforms.Compose([
                lambda file: Image.open(file),
                transforms.FiveCrop((self.image_dim, self.image_dim)),
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                lambda tensors: torch.stack(
                    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor) for tensor in tensors]
                )
            ])
            self.transform_catch = transforms.Compose([
                lambda file: Image.open(file),
                transforms.CenterCrop((self.image_dim, self.image_dim)),
                lambda crop: torch.stack([transforms.ToTensor()(crop) for _ in range(5)]),
                lambda tensors: torch.stack(
                    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor) for tensor in tensors]
                )
            ])
        else:
            self.transform = transforms.Compose([
                lambda file: Image.open(file),
                transforms.CenterCrop((self.image_dim, self.image_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        try:
            x = self.transform(self.files[idx])
        except ValueError as e:
            x = self.transform_catch(self.files[idx])
        return (x, self.labels[idx]) if self.return_label else x

    def __len__(self):
        return len(self.files)


def image_to_spec(image, spec_dim=224):
    image = np.array(image)

    spec = np.stack([np.fft.fftshift(np.fft.fft2(image[:, :, i])) for i in range(3)], axis=2)
    spec = 20 * np.log(np.abs(spec))
    spec = cv2.resize(spec, (spec_dim, spec_dim))
    spec_min = spec.min()
    spec_max = spec.max()
    spec = (spec - spec_min) / (spec_max - spec_min) * 255
    spec = spec.astype(np.uint8)
    return spec


class SpecDataset(Dataset):
    def __init__(self, image_dir, argument=True, image_dim=224, limit_num=None, return_label=True):
        self.argument = argument
        self.image_dim = image_dim
        self.return_label = return_label

        self.sub_dirs = sorted(os.listdir(image_dir))
        self.files = []
        self.labels = []
        for sub_dir in self.sub_dirs:
            temp = sorted(os.listdir(os.path.join(image_dir, sub_dir)))
            temp = [os.path.join(image_dir, sub_dir, file) for file in temp]
            temp = temp[:limit_num] if limit_num is not None else temp
            self.files.extend(temp)
            if self.return_label:
                self.labels.extend([int(sub_dir.split('_')[-1]) - 1 for _ in range(len(temp))])

        # self.transform = transforms.Compose([
        #     lambda file: Image.open(file),
        #     transforms.ToTensor()
        # ])

    def transform(self, file):
        spec_file = os.path.join('spec', file)
        if os.path.exists(spec_file):
            spec = Image.open(spec_file)
            spec = transforms.functional.to_tensor(spec)
        else:
            image = Image.open(file)
            spec = image_to_spec(image)

            os.makedirs(os.path.dirname(spec_file), exist_ok=True)
            Image.fromarray(spec).save(spec_file)

            spec = transforms.functional.to_tensor(spec)

        if self.argument:
            spec = transforms.RandomErasing()(spec)

        return spec

    def __getitem__(self, idx):
        try:
            x = self.transform(self.files[idx])
        except ValueError as e:
            x = self.transform_catch(self.files[idx])
        return (x, self.labels[idx]) if self.return_label else x

    def __len__(self):
        return len(self.files)
