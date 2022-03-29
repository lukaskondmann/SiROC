
import torch
import numpy as np
from torchvision import transforms
from data.custom_transforms import *
from data.helper_transforms import *
from data.utils import ONERA
from metrics import *
from method import *
import time, string
import math
from torchvision.utils import save_image
#from sklearn.metrics import auc,cohen_kappa_score,f1_score,accuracy_score,classification_report,recall_score,precision_score
import matplotlib.pyplot as plt
import cv2


# ----------
#  Run Method
# ----------
def main():
    locations = 'test.txt' # Either "train.txt" or "test.txt"
    dataset_source = 'Onera'
    path_dic = {
        'Onera': 'Onera',
        'Beirut': 'beirut_explosion_cd_ZKI',
        'Alpine':'lamarTrentoFireFeb2019',
        'Barrax':'barraxDataset',
        'DEN':'ev_2021_DEN/DEN_Binary_CD'}
    
    train_data = ONERA(root='/localhome/kond_lu/'+path_dic[dataset_source], csv_file=locations,
                       transform_op=ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)

    print('Load datasets from {}: train_set={}'.format('ONERA', len(train_data)))

    make_deterministic(10)
    #device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    plot = False 
    sample = False
    p = 0.2
    #searches = 1
    #for _ in range(searches):
    ensemble = True
    majority_voting = True  
    morph_opening = True
    profile = True
    plot_heatmap = False
    plot_confidence = False
    report_city_level = False
    max_neighborhood = 200
    mp_start,mp_stop,mp_step = 5,6,1
    exclusion = 0
    threshold = 'Otsu'
    splits = 27
    voting_threshold,otsu_factor = call_voting_parameters(dataset_source,morph_opening,ensemble,mp_start)

    print(max_neighborhood,exclusion,splits,mp_start)
    
    assert max_neighborhood > splits, "Maximum for split is the number of rows/columns in the neighborhood"
    
    print(threshold)
    c = time.time()
    TP_tot,TN_tot,FP_tot,FN_tot = 0,0,0,0
    for i, (batch, title) in enumerate(train_loader):
        out_title = ''.join([x for x in str(title) if x in string.ascii_letters])
        pre_img, post_img, label = batch['pre'], batch['post'], batch['gt']
        label = torch.where(label > torch.mean(label), torch.tensor(1), torch.tensor(0))
        assert label.max() == 1, "No Changing Pixels in the labels"
        assert label.min() == 0, "Only Changing Pixels in the labels"
            
        if ensemble == True:
            neighborhood,ex = split_neighborhood_uniform(max_neighborhood,splits,exclusion)[0]
            change_map = obtain_change_map(pre_img, post_img, neighborhood=neighborhood,excluded=ex,sample=sample,p=p)

            for neighborhood,ex in split_neighborhood_uniform(max_neighborhood,splits,exclusion)[1:]:
                change_map =torch.cat((change_map,obtain_change_map(pre_img, post_img, neighborhood=neighborhood,excluded=ex,sample=sample,p=p)),dim=0) 
        
            # Take absolute value of change signal
            change_map = torch.abs(change_map)
            
            # Average across spectral bands 
            change_map = change_map.mean(dim=1)
            
            if majority_voting == True:
            # Apply threshold for each NN individually 
                l = 0
                for j in change_map:
                    apply_threshold(change_map,j,threshold,l,otsu_factor)
                    if morph_opening == True:
                        if profile == True: 
                            morph_profile = torch.zeros_like(change_map[l,:,:])
                            for kernel in range(mp_start,mp_stop,mp_step):
                                opening = torch.tensor(cv2.morphologyEx(np.uint8(change_map[l,:,:].squeeze().numpy()), cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8)))
                                morph_filter = torch.tensor(cv2.morphologyEx(np.uint8(opening), cv2.MORPH_OPEN,np.ones((kernel,kernel),np.uint8)))
                                morph_profile += morph_filter
                            change_map[l,:,:] = morph_profile/math.ceil(((mp_stop-mp_start)/mp_step))
                        else: 
                            change_map[l,:,:] = torch.tensor(cv2.morphologyEx(np.uint8(change_map[l,:,:].squeeze().numpy()), cv2.MORPH_CLOSE,np.ones((close_x,close_y),np.uint8)))
                            change_map[l,:,:] = torch.tensor(cv2.morphologyEx(np.uint8(change_map[l,:,:].squeeze().numpy()), cv2.MORPH_OPEN,np.ones((open_x,open_y),np.uint8)))
                           
                    l += 1
                    
                    
                print('Max:',change_map.max())
                print('Min:',change_map.min())
                assert change_map.max() == 1
                change_map = change_map.mean(dim=0)
                
                if plot_heatmap == True:
                    plt.imshow(change_map.numpy())
                    plt.axis('off')
                    #plt.savefig('Plots/Heatmaps/'+dataset_source+'_Confidence_Heatmap_'+str(out_title.capitalize())+'.pdf', bbox_inches='tight')
                    #plt.imsave('Plots/Heatmaps/'+dataset_source+'_Confidence_Heatmap_'+str(out_title.capitalize())+'.pdf',change_map.numpy())
                    plt.show()
                if plot_confidence == True:
                    plot_confidence_scores(change_map,splits,voting_threshold,label,out_title)
                    
                
                change_map = torch.where(abs(change_map) >= (voting_threshold), torch.tensor(1), torch.tensor(0))
            
            elif majority_voting == False: 
            # Take mean of different ensembles 
                change_map = torch.unsqueeze(change_map.mean(dim=0),0)
                print('Shape CM',change_map.shape)
                change_map = torch.squeeze(change_map)
                print('Shape CM',change_map.shape)
                #change_map = torch.where(abs(change_map) > threshold, torch.tensor(1), torch.tensor(0))
                if threshold == 'Otsu':
                    otsu,otsu_cm = cv2.threshold(np.array(abs(change_map.numpy()* 255), dtype = np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    thresh = otsu/255
                else:
                    thresh = threshold 
                change_map = torch.where(abs(change_map) > (thresh), torch.tensor(1), torch.tensor(0))

            
        if ensemble == False:
            ex = exclusion
            change_map = obtain_change_map(pre_img, post_img, neighborhood=neighborhood,excluded=exclusion,sample=sample,p=p)
            change_map = torch.squeeze(change_map.mean(dim=1))    
            
            
            plt.imshow(change_map.numpy(),cmap='gray')
            plt.title('Activation Map')
            plt.show()
            
            t= cv2.threshold(np.array(abs(change_map.numpy()* 255), dtype = np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
            change_map = torch.where(abs(change_map) > (t*otsu_factor/255), torch.tensor(1), torch.tensor(0))
            if morph_opening == True:
                change_map = torch.tensor(cv2.morphologyEx(np.uint8(change_map.squeeze().numpy()), cv2.MORPH_CLOSE,np.ones((close_x,close_y),np.uint8)))
                change_map = torch.tensor(cv2.morphologyEx(np.uint8(change_map.squeeze().numpy()), cv2.MORPH_OPEN,np.ones((close_x,close_y),np.uint8)))
                
    
        print(change_map.shape)
        print(label.shape)
        #plt.axis('off')

        if plot == True: 
            plt.imshow(change_map.numpy(),cmap='gray')
            plt.axis('off')
            plt.title('Final Change Map')
            #plt.imsave('Plots/Change_Maps/'+dataset_source+'_Change_Map_'+str(out_title.capitalize())+'_MP='+str(morph_opening)+'.pdf',change_map.numpy(),cmap='gray')
            plt.show()
            
            plt.imshow(torch.squeeze(label).numpy(),cmap='gray')
            plt.axis('off')
            plt.title('Labels')
            #plt.imsave('Plots/Labels/'+dataset_source+'_Labels_'+str(out_title.capitalize())+'.pdf',torch.squeeze(label).numpy(),cmap='gray')
            plt.show()
        
        
            print(pre_img.shape)
            pre_RGB = cv2.cvtColor(torch.squeeze(pre_img).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            post_RGB = cv2.cvtColor(torch.squeeze(post_img).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            plt.imshow(pre_RGB)
            plt.show()
            plt.imshow(post_RGB)
        
            #plt.imsave('Plots/'+dataset_source+'_Pre_'+str(out_title.capitalize())+'.pdf',pre_RGB)
            #plt.imsave('Plots/'+dataset_source+'_Post_'+str(out_title.capitalize())+'.pdf',post_RGB)
            plt.show()
                
    
        #acc, change_acc, no_change_acc,TP,TN,FP,FN = get_accuracy(change_map.numpy(),label.reshape(label.shape[2],label.shape[3]).numpy())
        TP,TN,FP,FN = get_accuracy(change_map.numpy(),label.reshape(label.shape[2],label.shape[3]).numpy())
        
        if report_city_level == True:
            print('City Number:',i)
            print('Sentivitity/Recall:',TP/(TP+FN))   
            print('Specificity',TN/(TN+FP))  
            print('Precision',TP/(TP+FP)) 
        
        # update total numbers with current city
        TP_tot+=TP
        TN_tot+=TN
        FP_tot+=FP
        FN_tot+=FN
        
        if dataset_source == 'Barrax' or dataset_source == 'Alpine':
            print(out_title.capitalize())
            print('Resetting counts because its not different locations, just different inputs')
            recall = TP/(TP+FN)
            specificity = TN/(TN+FP)
            precision = TP/(TP+FP)
            print('Sentivitity/Recall:',round(recall,4))  
            print('Specificity:',round(specificity,4))
            print('Precision:',round(precision,4))  
            print('F1:',round((2 * precision * recall) / (precision + recall),4))

    
    #recall = TP_tot/(TP_tot+FN_tot)
    #specificity = TN_tot/(TN_tot+FP_tot)
    #precision = TP_tot/(TP_tot+FP_tot)

    print('One model takes (whole dataset):',time.time()-c)
    if dataset_source == 'Onera' or dataset_source == 'Beirut':
        print(TP_tot,TN_tot,FP_tot,FN_tot)
        if dataset_source == 'Onera' and locations == 'test.txt':
            assert TP_tot+TN_tot+FP_tot+FN_tot == 3077936, "Total # pixels for OSCD must equal 3077936"
        recall = TP_tot/(TP_tot+FN_tot)
        specificity = TN_tot/(TN_tot+FP_tot)
        precision = TP_tot/(TP_tot+FP_tot)
        print('Sentivitity/Recall:',round(recall,4))  
        print('Specificity:',round(specificity,4))
        print('Precision:',round(precision,4))  
        print('F1:',round((2 * precision * recall) / (precision + recall),4))

# %%

if __name__ == '__main__':

    main()


