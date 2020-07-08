import glob
import os

import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.pth'

    __constants__ = ['resnet', 'fc', 'head_lab', 'head_unl']

    def __init__(self):
        super(Model, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # print(resnet)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU()
        )

        self.head_lab = nn.Sequential(nn.Linear(256, 10))
        self.head_unl = nn.Sequential(nn.Linear(256, 4))

    def forward(self, x):
        x = self.resnet(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        lab_logits = self.head_lab(x)
        unl_logits = self.head_unl(x)

        return lab_logits, unl_logits

    def store(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(os.path.split(path_to_model)[-1][6:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(os.path.split(path_to_checkpoint_file)[-1][6:-4])
        return step

class Model2(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.pth'

    __constants__ = ['resnet', 'fc', 'head_lab', 'head_unl']

    def __init__(self):
        super(Model2, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # print(resnet)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU()
        )

        self.head_lab = nn.Sequential(nn.Linear(256, 10))
        self.head_unl = nn.Sequential(nn.Linear(256, 6))

    def forward(self, x):
        x = self.resnet(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        lab_logits = self.head_lab(x)
        unl_logits = self.head_unl(x)

        return lab_logits, unl_logits

    def store(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(os.path.split(path_to_model)[-1][6:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)
        path_to_checkpoint_file = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(os.path.split(path_to_checkpoint_file)[-1][6:-4])
        return step

