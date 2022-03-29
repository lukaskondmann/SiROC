import torch
import numpy as np
from torchvision import transforms
from data.custom_transforms import *
from data.helper_transforms import *
from data.utils import ONERA
from metrics import *
import time, string
import math
import shutil
from torchvision.utils import save_image
import os
from sklearn.metrics import auc,cohen_kappa_score,f1_score,accuracy_score,classification_report,recall_score,precision_score
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pandas as pd
import cv2
import rasterio 
import rasterio.plot
import glob

paths =  glob.glob('/mnt/ushelf/datasets/landcover_dynamicearth/bundle/planet/*/*/1417_3281_13/PF-SR/2018-01-01.tif')
#paths = glob.glob('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/*/PF-SR/2018-01-01.tif')
print(paths)

for path in paths: 
    ## Open GeoTiffs
    pre_path = path
    #post_path = "C:/Users/kond_lu/Documents/data/Dynamic_Earth_Net_Binary/1487_3335_13/11N-114W-31N-L3H-SR-2019_12_01.tif"
    post_path = path[:-14] + '2019-12-01.tif' 
    #cube = path.split("\\")[1]
    cube = path.split("/")[-3]
    print(post_path)
    print(cube)
    pre = rasterio.open(pre_path).read()
    post = rasterio.open(post_path).read()

    index = [2,1,0] # Red, Green, Blue Bands 
    rgb_pre = pre[index,:,:]/10000
    rgb_post = post[index,:,:]/10000
    
    #print(rgb_pre.min(),rgb_pre.max(),rgb_post.min(),rgb_post.max())
  
    rgb_pre /= rgb_pre.max()
    rgb_post /= rgb_post.max() 
    #print(rgb_pre.min(),rgb_pre.max(),rgb_post.min(),rgb_post.max())
    
    os.makedirs('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/'+cube +'/'+'/pair/', exist_ok=True)

    plt.imshow(np.moveaxis(rgb_pre[:3,:,:],0,-1))
    plt.imsave('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/'+cube +'/' +'/pair/'+'img1.png',np.moveaxis(rgb_pre[:3,:,:],0,-1))
    plt.show()
    plt.imshow(np.moveaxis(rgb_post[:3,:,:],0,-1))
    plt.imsave('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/'+cube +'/'+'/pair/'+'img2.png',np.moveaxis(rgb_post[:3,:,:],0,-1))
    plt.show()


