import torch
import numpy as np
import cv2

 

def obtain_change_map(pre, post, neighborhood, excluded=0, lamb=1e-5,sample=False,p=0.1):
    '''This function gets the change map for a pair of pre/post input images
    Predictions at the edges are zero-padded, i.e. only pixels inside the original pre image matter.'''
    #get post shapen
    
    B, C, H_pre, W_pre = pre.shape
    _, _, H_post, W_post = post.shape

    # Generate array to save predictions in
    padded_pre = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_pre[:, :, neighborhood:H_pre + neighborhood, neighborhood:W_pre + neighborhood] = pre
    padded_post = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_post[:, :, neighborhood:H_post + neighborhood, neighborhood:W_post + neighborhood] = post

    num_neighbors = (2 * neighborhood + 1) ** 2 - (2 * excluded + 1)**2
    #np.randint(num_neigh)

    pre_response = padded_pre ** 2
    post_response = padded_pre * padded_post
    pre_sum = torch.zeros(post.shape)
    post_sum = torch.zeros(post.shape)

    for x_patch in range(-neighborhood, neighborhood + 1):
											   
        for y_patch in range(-neighborhood, neighborhood + 1):
            if abs(x_patch) <= excluded or abs(y_patch) <= excluded:
                continue
										 
																				 
																				 

            if sample == True: 
                #weights_rand = torch.as_tensor(np.random.lognormal(0, 0.125, post.shape))
                #weights_rand = torch.as_tensor(np.random.randint(0, 2, post.shape)) + 1e-5
                # With weights 
                weights_rand = torch.as_tensor(np.random.choice(2, post.shape,p=[p,1-p])) + 1e-5

                pre_sum += weights_rand * pre_response[:, :,
                                           y_patch + neighborhood:H_pre + y_patch + neighborhood,
                                           x_patch + neighborhood:W_pre + x_patch + neighborhood]
    
                post_sum += weights_rand * post_response[:, :,
                                            y_patch + neighborhood:H_post + y_patch + neighborhood,
                                            x_patch + neighborhood:W_post + x_patch + neighborhood]
                
            else: 
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
    elif threshold == 'Gaussian':
        img = np.int8(np.array(j*255).ravel())
        assert np.isnan(img).any() == False, "NaNs in array"
        gmm = GaussianMixture(n_components = 2)
        gmm = gmm.fit(X=np.expand_dims(img,1))
        t = np.mean(gmm.means_)
        change_map[l,:,:] = torch.where(abs(j) > (t/255), torch.tensor(1), torch.tensor(0))
    else:
        assert threshold in ['Otsu','Triangle','Gaussian'], "Thresholding not identified"
    return change_map

def get_conv(neighborhood,excluded,device):
    sum_conv = torch.nn.Conv2d(3, 1, neighborhood*2,
                               padding=0,
                               bias=False)
    sum_conv.eval()
    sum_conv = sum_conv.to(device)
    sum_conv.weight.data.fill_(1)
    delta = neighborhood - excluded
    sum_conv.weight.data[:, :, delta:-delta, delta:-delta] = 0.0
    return sum_conv

def obtain_change_map_conv(pre, post, neighborhood, sum_conv):
    '''This function gets the change map for a pair of pre/post input images
    Predictions at the edges are zero-padded, i.e. only pixels inside the original pre image matter.'''
    #get post shapen

    pre = pre.cuda(non_blocking=True)
    post = post.cuda(non_blocking=True)
    nb = neighborhood
    pad = (nb, nb-1, nb, nb-1)
    pre = torch.nn.functional.pad(input=pre, pad=pad, mode='constant', value=0)
    post = torch.nn.functional.pad(input=post, pad=pad, mode='constant', value=0)

    # print(sum_conv.weight.shape)
    with torch.no_grad():
        pre_sums = sum_conv(pre).squeeze()
        post_sums = sum_conv(post).squeeze()
        pre = pre[:,:, nb:-(nb-1),nb:-(nb-1)]
        post = post[:,:,nb:-(nb-1),nb:-(nb-1)]
        assert pre.shape[-2:] == pre_sums.shape[-2:], f'Sizes dont match: {pre.shape} and {pre_sums.shape}!'
        coefs = (pre  / pre_sums.unsqueeze(1))
        preds = coefs*post_sums.unsqueeze(1)
        change_map = preds - post

    return change_map.cpu()
