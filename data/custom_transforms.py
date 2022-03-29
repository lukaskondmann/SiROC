import random
import cv2
import numpy as np
import torch
from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = sample[elem]            
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            tmp = np.array(tmp, dtype=np.float32).transpose((2, 0, 1))
            tmp /= 255.0
            sample[elem] = torch.from_numpy(tmp)

        return sample


class Resize(object):
    def __init__(self, resize=(512, 512)):
        self.resize = resize
    def __call__(self, sample):

        for elem in sample.keys():
            #if elem == 'gt':
             #   continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, self.resize, interpolation=flagval)

            sample[elem] = tmp

        return sample
