import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import torchvision.transforms.functional as TF
import random


class RandRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class AngleRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class HorizontalFlip:
    def __init__(self):
        pass

    def __call__(self, x):
        return TF.hflip(x)


class VerticalFlip:
    def __init__(self):
        pass

    def __call__(self, x):
        return TF.rotate(TF.hflip(x), 180)


def make_transform(tranform_fn):
    return transforms.Compose(
        [tranform_fn,
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])


class LabDataset(data.Dataset):
    def __init__(self, set):
        self.lab_set = set
        self.lab_trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    def __getitem__(self, i):
        return self.lab_trans(self.lab_set[i][0]), self.lab_set[i][1]

    def __len__(self):
        return len(self.lab_set)


class RotDataset(data.Dataset):
    def __init__(self, set, angle):
        """
        :param set: unlabelled dataset
        :param angle: should be 0, 1, 2, 3 stand for 0, 90, 180, 270.
        """
        self.set = set
        self.angle = angle
        self.tranfn = make_transform(AngleRotationTransform(angle * 90))

    def __getitem__(self, i):
        return self.tranfn(self.set[i][0]), torch.tensor(self.angle)

    def __len__(self):
        return len(self.set)


class FlipDataset(data.Dataset):
    def __init__(self, set, type):
        """
        :param set: unlabelled dataset
        :param type: should be 4, 5 stand for horizontal, vertical flip
        """
        self.set = set
        self.type = type
        if type == 4:
            self.tranfn = make_transform(HorizontalFlip())
        else:
            self.tranfn = make_transform(VerticalFlip())            

    def __getitem__(self, i):
        return self.tranfn(self.set[i][0]), torch.tensor(self.type)

    def __len__(self):
        return len(self.set)

