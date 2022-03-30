import os
import numpy as np
import cv2
from matplotlib import pyplot
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
from data.custom_transforms import ToTensor, Resize
from data.helper_transforms import *

class ONERA(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform_op=None):
        self.root = root
        self.transform_op = transform_op
        self.isTrain = 1 if csv_file == 'train.txt' else 0
        # Load image list
        csv_path = os.path.join(root, 'splits', csv_file)

        with open(csv_path, 'r') as f:
            lines = f.readlines()
            pre_img = []
            post_img = []
            labels = []
            img_list = []
            if self.isTrain:
                for line in lines:
                    images = os.listdir(os.path.join(root, 'Images', line.strip(), 'pair'))
                    images_path = list(map(lambda x: os.path.join('Images', line.strip(), 'pair',  x), images))
                    pre_img.append(images_path[0])
                    post_img.append(images_path[1])
                    lab = np.sort(os.listdir(os.path.join(root, 'TrainLabels', line.strip(), 'cm')))
                    lab_path = list(map(lambda x: os.path.join('TrainLabels', line.strip(), 'cm', x), lab))
                    lab_path = [x for x in lab_path if 'cm.png' in x] # Remove irrelevant entries
                    labels.append(lab_path[0])
                    img_list.append(line.strip())

            else:
                for line in lines:
                    images = os.listdir(os.path.join(root, 'Images', line.strip(), 'pair'))
                    images_path = list(map(lambda x: os.path.join('Images', line.strip(), 'pair', x), images))
                    pre_img.append(images_path[0])
                    post_img.append(images_path[1])
                    lab = np.sort(os.listdir(os.path.join(root, 'TestLabels', line.strip(), 'cm')))
                    lab_path = list(map(lambda x: os.path.join('TestLabels', line.strip(), 'cm', x), lab))
                    lab_path = [x for x in lab_path if 'cm.png' in x] # Remove irrelevant entries
                    labels.append(lab_path[0])
                    img_list.append(line.strip())

        assert (len(labels) == len(pre_img) == len(post_img))

        self.num = len(pre_img)
        self.pre_img = pre_img
        self.post_img = post_img
        self.labels = labels
        self.img_list = img_list

        print('Done initializing Dataset')

    def __getitem__(self, idx):
        img1, img2, gt = self.make_img_gt_pair(idx)
        sample = {'pre':img1, 'post':img2, 'gt':gt}
        title = self.img_list[idx]
        if self.transform_op is not None:
            sample = self.transform_op(sample)
        return sample, str(title)

    def make_img_gt_pair(self, idx):
        #Make the image1-image2-ground-truth pair

        img1 = cv2.imread(os.path.join(self.root, self.pre_img[idx]))
        img2 = cv2.imread(os.path.join(self.root, self.post_img[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.root, self.labels[idx]), 0)
        else:
            gt = np.zeros(img1.shape[:-1], dtype=np.uint8)
            
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = np.array(img1, dtype=np.float32)
        img2 = np.array(img2, dtype=np.float32)

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            # gt = gt / np.max([gt.max(), 1e-8]) 
           
        return img1, img2, gt


    def __len__(self):
        return self.num

if __name__ == '__main__':

    custom_transforms = transforms.Compose([Resize(resize=(512, 512)), ToTensor()])
    train_data = ONERA(root='C:/Users/kond_lu/Documents/data/Onera', csv_file='test.txt',
                           transform_op=custom_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
    for i, (data, title) in enumerate(train_loader):
        pre_img, pos_img, gt = data['pre'], data['post'], data['gt']
        #pre = [B, C, H, W]
        print ('label shape', gt.shape)
        print ('label max', torch.max(gt))
        print ('label min', torch.min(gt))

        print ('img shape', pre_img.shape)
        print (torch.max(pre_img))
        print (torch.min(pre_img))

        fig, axs = pyplot.subplots(nrows=3, ncols=1, figsize=(10, 10))
        #pyplot.subplot(3, 1, 1)

        axs[0].imshow(im_normalize(tens2image(pre_img)))
        print ('after', pre_img.max())
        print ('after', pre_img.min())
        axs[1].imshow(im_normalize(tens2image(pos_img)))
        axs[2].imshow(tens2image(gt))
        pyplot.show(block=True)

        if i == 5:
           break

