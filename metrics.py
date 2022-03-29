import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch


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
    else:
        print('Invalid data source')
        
    return voting_threshold,otsu_factor
