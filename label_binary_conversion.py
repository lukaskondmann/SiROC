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


#paths = glob.glob('/mnt/ushelf/datasets/landcover_dynamicearth/labels/*/Labels/Raster/*/*-*-*-L3H-SR-2018*01*01.tif')
paths = glob.glob('/mnt/ushelf/datasets/landcover_dynamicearth/labels/*/Labels/Raster/*/11N-117W-33N-L3H-SR-2018*01*01.tif')

print(paths)
for path in paths: 
    ## Open GeoTiffs
    pre_path = path
    #post_path = "C:/Users/kond_lu/Documents/data/Dynamic_Earth_Net_Binary/1487_3335_13/11N-114W-31N-L3H-SR-2019_12_01.tif"
    # catch label naming inconsistency 
    if '2018_01_01.tif' in pre_path:
        post_path = path[:-14] + '2019_12_01.tif' 
    elif '2018-01-01.tif' in pre_path:
        post_path = path[:-14] + '2019-12-01.tif' 
    cube = path.split("/")[-5][:-4] #previous version with local path  
    print(post_path)
    print(cube)
    pre = rasterio.open(pre_path).read()
    post = rasterio.open(post_path).read()


    #plt.imshow(pre[:3,:,:].reshape(1024,1024,3))
    #plt.imshow(post[:3,:,:].reshape(1024,1024,3))
    #plt.show()

    ## Calculate Changing Pixels 
    #assert pre.shape==(7, 1024, 1024),"Dimension of Labels are wrong"
    diff = np.sum(abs(pre-post)>0,axis=0)

    # Visualize images to compare labeled changes 
    pre = plt.imread('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/'+cube +'/' +'/pair/'+'img1.png')
    post = plt.imread('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/Images/'+cube +'/'+'/pair/'+'img2.png')
    plt.imshow(pre)
    plt.show()
    plt.imshow(post)
    plt.show()

    ## Plot and save result
    plt.imshow(diff,cmap='gray')
    os.makedirs('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/TrainLabels/'+cube +'/cm/', exist_ok=True)
    plt.imsave('/localhome/kond_lu/ev_2021_DEN/DEN_Binary_CD/TrainLabels/'+cube +'/cm/'+'cm.png',diff,cmap='gray')
    plt.show()
