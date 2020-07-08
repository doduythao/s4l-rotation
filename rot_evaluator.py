import torch
import numpy as np
from torch.utils.data import DataLoader

from rot_dataset import LabDataset


class Evaluator(object):
    def __init__(self, val_set):
        self._loader = DataLoader(LabDataset(val_set), batch_size=128, shuffle=False)

    def evaluate(self, model):
        num_correct1 = 0
        num_correct5 = 0

        with torch.no_grad():
            for images, ys in self._loader:
                images, ys = images.cuda(), ys.cuda()
                lab_y_log, _ = model.eval()(images)
                lab_y_pred = lab_y_log.max(1)[1]
                _, indices = lab_y_log.topk(5, dim=1)
                
                predicts = indices.cpu().detach().numpy()
                ys_np = ys.cpu().detach().numpy()
                temp_correct = 0
                for i in range(len(ys_np)):
                    if ys_np[i] in predicts[i]:
                        temp_correct +=1

                num_correct5 += temp_correct
                num_correct1 += (lab_y_pred.eq(ys)).cpu().sum()

        dataset_size = len(self._loader.dataset)
        accuracy1 = num_correct1.item() / dataset_size
        accuracy5 = num_correct5 / dataset_size
        return accuracy1, accuracy5
