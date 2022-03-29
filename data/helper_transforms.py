import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import random
import torch


def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def im_normalize(im):
    #Normalize image
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def deep_iter(data, ix=tuple()):
    try:
        for i, element in enumerate(data):
            yield from deep_iter(element, ix + (i,))
    except:
        yield ix, data

#Add seeding
def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important also