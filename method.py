import torch
import numpy as np
import cv2

 
def obtain_change_map(pre, post, neighborhood, excluded=0):
    '''This function gets the change map for a pair of pre/post input images
    Predictions at the edges are zero-padded, i.e. only pixels inside the original pre image matter.'''
    
    B, C, H_pre, W_pre = pre.shape
    _, _, H_post, W_post = post.shape

    # Generate array to save predictions in
    padded_pre = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_pre[:, :, neighborhood:H_pre + neighborhood, neighborhood:W_pre + neighborhood] = pre
    padded_post = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_post[:, :, neighborhood:H_post + neighborhood, neighborhood:W_post + neighborhood] = post

    num_neighbors = (2 * neighborhood + 1) ** 2 - (2 * excluded + 1)**2

    pre_response = padded_pre ** 2
    post_response = padded_pre * padded_post
    pre_sum = torch.zeros(post.shape)
    post_sum = torch.zeros(post.shape)

    # Iterate over patches
    for x_patch in range(-neighborhood, neighborhood + 1):
											   
        for y_patch in range(-neighborhood, neighborhood + 1):
            if abs(x_patch) <= excluded or abs(y_patch) <= excluded:
                continue

            pre_sum += pre_response[:, :,
                                           y_patch + neighborhood:H_pre + y_patch + neighborhood,
                                           x_patch + neighborhood:W_pre + x_patch + neighborhood]
    
            post_sum += post_response[:, :,
                                        y_patch + neighborhood:H_post + y_patch + neighborhood,
                                        x_patch + neighborhood:W_post + x_patch + neighborhood]						 

    post_pred = pre * post_sum / pre_sum
    change_map = torch.abs(post_pred - post)
    #print ('change map shape', change_map.shape)
    return change_map



def apply_threshold(change_map,j,threshold,l,otsu_factor):
    if threshold == 'Otsu':
        img = np.int8(np.array(j*255).ravel())
        assert np.isnan(img).any() == False
        t= cv2.threshold(np.array(abs(j.numpy()* 255), dtype = np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
        change_map[l,:,:] = torch.where(abs(j) > (t*otsu_factor/255), torch.tensor(1), torch.tensor(0))
    elif threshold == 'Triangle':
        t= cv2.threshold(np.array(abs(j.numpy()* 255), dtype = np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE )[0]
        change_map[l,:,:] = torch.where(abs(j) > (t*0.5*otsu_factor/255), torch.tensor(1), torch.tensor(0))
    else:
        assert threshold in ['Otsu','Triangle','Gaussian'], "Thresholding not identified"
    return change_map
