import numpy as np
import glob
import os
print(os.getcwd())
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,precision_score,recall_score, auc
from sklearn.metrics import auc,cohen_kappa_score,f1_score,accuracy_score,classification_report,recall_score,precision_score
import pickle
import pandas as pd 
import torch

def evaluate_change_map(change_map, label, threshold):
    '''
    Input: Change Map, Threshold
    Output: Overall accuracy, In-class accuracy.

    This function takes a change map as input, maps the change map into change/no change based on the threshold
    and evaluates this based on the ground truth. Class labels are 2 (change) and 1 (no change) to be comparable with ONERA labels.'''

    # Average CM across RGB
    change_map = change_map.data
    label = label.data
    change_map = np.average(change_map, axis=2)

    # Apply threshold
    change_class = np.zeros_like(change_map)
    change_class[abs(change_map) > threshold] = 2
    change_class[abs(change_map) <= threshold] = 1
    assert change_class.all() != 0

    scores = get_accuracy(change_class, label)
    return scores


def call_voting_parameters(dataset_source,morph_opening,ensemble,mp_start):
    voting_threshold = 0.5
    if dataset_source == 'Onera':
        if morph_opening == True:
            otsu_factor = 1
            voting_threshold= 0.5
        else: 
            if ensemble == True:
               voting_threshold = 0.35 
               otsu_factor = 0.75
            else:
                otsu_factor = 0.65
    elif dataset_source == 'Beirut':
        if morph_opening == True:
            otsu_factor = 0.8
            voting_threshold= 0.5
            if mp_start == 10:
               otsu_factor = 1 
        else: 
            if ensemble == True:
               voting_threshold = 0.35 
               otsu_factor = 0.75
            else:
                otsu_factor = 0.7
                
    #elif dataset_source == 'California':
     #   mp_start,mp_stop,mp_step = 3,4,1
    elif dataset_source == 'Alpine':
        if morph_opening == True:
            otsu_factor = 1.8
            voting_threshold= 0.5
        else: 
            if ensemble == True:
               voting_threshold = 0.5
               otsu_factor = 1.7
            else:
                otsu_factor = 1.7
    elif dataset_source == 'Barrax':
        if morph_opening == True:
            otsu_factor = 1
            voting_threshold= 0.5
        else: 
            if ensemble == True:
               voting_threshold = 0.5 
               otsu_factor = 0.95
            else:
                otsu_factor = 1

    elif dataset_source == 'DEN':
        if morph_opening == True:
            otsu_factor = 1
            voting_threshold= 0.2
        else: 
            if ensemble == True:
               voting_threshold = 0.5 
               otsu_factor = 0.95
            else:
                otsu_factor = 1            
    
    else:
        print('Invalid data source')
        
    return voting_threshold,otsu_factor


def get_pixel_predictions(x, y, pre, padded_pre, padded_post, post_pred, bands=3, neighborhood=2, excluded=0, lamb=0):
    '''This function models a pixel (x,y) based on the pixel values of its neighborhood with a linear regression in t-1
    and obtains predictions for the same pixel (x,y) in t based on the surrounding pixels in t and the coefficients
    of the regression. By default, the center pixel is not used for the regression and further pixels can be exluded by
    setting excluded > 0. Neighborhood determines how many pixels are used for the regression and excluded how many levels
    of neighborhood to the center are excluded (e.g excluded = 1 => 8 directly adjacent pixels to center are not used)
    '''

    # Get patch of relevant pixels
    patch = np.array(padded_pre[x:x + 1 + 2 * neighborhood, y:y + 1 + 2 * neighborhood, :])
    # Make center pixel zero (don't use label)
    patch[patch.shape[0] - neighborhood - excluded - 1:patch.shape[0] - neighborhood + excluded,
    patch.shape[0] - neighborhood - excluded - 1:patch.shape[0] - neighborhood + excluded, :] = 0
    # Iterate over bands
    for band in np.arange(bands):
        # Obtain coefficients
        end = patch[:, :, band].reshape(1, patch.shape[0] * patch.shape[1])
        dep = pre[x, y, band].reshape(1, -1)
        # coeff = np.linalg.lstsq(end , dep, rcond=None)[0]
        coeff = np.linalg.lstsq(end + lamb * np.identity(end.shape[0]), dep, rcond=None)[0]
        # Get prediction of new pixel
        patch_post = np.array(padded_post[x:x + 1 + 2 * neighborhood, y:y + 1 + 2 * neighborhood, band])
        prediction = np.sum(np.multiply(patch_post.reshape(patch_post.shape[0] * patch_post.shape[1], 1),
                                        coeff))  # This returns the same as the statsmodels sm.OLS function

        # Save prediction in new_array
        post_pred[x, y, band] = prediction


def get_accuracy(change, label):
    '''This functions measures overall and within class accuracy
    for binary change detection results'''
    assert change.shape == label.shape, "Labels and predicted change mask do not have the same size"
    TP = np.size(change[(change == 1) & (label == 1)])
    TN = np.size(change[(change == 0) & (label == 0)])
    FP = np.size(change[(change == 1) & (label == 0)])
    FN = np.size(change[(change == 0) & (label == 1)])
    return (TP,TN,FP,FN)

def split_neighborhood_uniform(neighborhood,splits,excluded):
    '''This function takes in a maximum neighborhood size and splits it
    in equally distanced neighborhoods for the ensemble method. 
    Note that the number of points in these neighborhoods is not identical'''

    splitted_neighborhoods = []
    step = (neighborhood-excluded) // splits 
    for i in range(1,splits+1):
        local_neighborhood = step * i + excluded
        exclusion = step * (i-1) + excluded
        splitted_neighborhoods.append((local_neighborhood,exclusion))
    return splitted_neighborhoods 

def plot_confidence_scores(change_map,splits,voting_threshold,label,out_title):
    ''' This function plots the Precision of the classification as a function
    of the confidence in the predictions. If the uncertainties are well
    calibrated, this should be increasing.    
    '''
    change_map_sum = torch.round(change_map * splits)
    confidence = torch.abs(change_map_sum - torch.round(torch.tensor(splits/2)))
    prediction = torch.where(abs(change_map) >= (voting_threshold), torch.tensor(1), torch.tensor(0))
    confidence_scores = []
    for m in range(int(torch.max(confidence))):
        prediction_batch = prediction[confidence == m]
        labels_batch = label.squeeze()[confidence == m]
        report = classification_report(labels_batch.numpy().flatten(),prediction_batch.numpy().flatten(),output_dict=True)
        precision,recall = report['1']['precision'], report['1']['recall']
        confidence_scores.append((precision,recall))
    precision = [x[0] for x in confidence_scores]
    plt.scatter(range(int(torch.max(confidence))),precision)
    plt.xlabel('Model Confidence')
    plt.ylabel('Precision')
    #plt.title('Confidence-Performance Curve' + ' ' +str(out_title.capitalize()))
    plt.title('Confidence-Performance Curve')
    plt.savefig('Plots/Confidence/Confidence_Performance'+str(out_title.capitalize())+'.pdf')
    plt.show()

