## SiROC - Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images

Implementation of the IEEE TGRS paper [Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images]( https://ieeexplore.ieee.org/document/9627707) by Kondmann et al. (2021)

![](sample_SiROC.png)

### Abstract
Detecting changes on the ground in multitemporal Earth observation data is one of the key problems in remote sensing. 
In this paper, we introduce Sibling Regression for Optical Change detection (SiROC), an unsupervised method for change 
detection in optical satellite images with medium and high resolution. SiROC is a spatial context-based method that models 
a pixel as a linear combination of its distant neighbors. It uses this model to analyze differences in the pixel and its spatial 
context-based predictions in subsequent time periods for change detection. We combine this spatial context-based change detection 
with ensembling over mutually exclusive neighborhoods and transitioning from pixel to object-level changes with morphological operations. 
SiROC achieves competitive performance for change detection with medium-resolution Sentinel-2 and high-resolution Planetscope imagery 
on four datasets. Besides accurate predictions without the need for training, SiROC also provides a well-calibrated uncertainty of its predictions.


### Data 

Get the OSCD dataset from [IEEE Dataport](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection). 
Unfortunately, we do not have permission to share the three smaller datasets involved in the paper as we do not own them. 
To use our data loader, move the train.txt and test.txt files from the image directory into a separate folder "splits" in the main folder of the dataset. In the end, your structure should look like this:

├── Onera Dataset                   
│   ├── Images    
│   ├── splits  
│   ├── TestLabels
│   ├── TrainLabels

### Requirements 

```
pip install -U -r requirements.txt
```
### Evaluation of SiROC

Once you downloaded the OSCD dataset, adjust data_path and out_dir in train.py and simply run it. Parameters are set to the paper defaults.

### Reference  

If you use our method, please cite:

```
@article{kondmann2021spatial,
  title={Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images},
  author={Kondmann, Lukas and Toker, Aysim and Saha, Sudipan and Sch{\"o}lkopf, Bernhard and Leal-Taix{\'e}, Laura and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```
