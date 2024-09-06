import torch
import sys
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import cv2
import csv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

def cal_loss(fs_list, ft_list,norm):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2) if norm else fs
        ft_norm = F.normalize(ft, p=2) if norm else ft
 
        f_loss = 0.5 * (ft_norm - fs_norm) ** 2
        f_loss = f_loss.sum() / (h * w)
        t_loss += f_loss

    return t_loss / N


@torch.no_grad()
def cal_anomaly_maps(fs_list, ft_list, out_size,norm):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2) if norm else fs
        ft_norm = F.normalize(ft, p=2) if norm else ft

        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)

        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(
            a_map, size=out_size, mode="bilinear", align_corners=False
        )
        anomaly_map += a_map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map

def th_img(img,threshold):
    max=np.max(img)
    ret, thresh = cv2.threshold(img, threshold*max, max, cv2.THRESH_BINARY)
    return thresh

def write_in_csv(csv_path,line_array):
    '''
    line array should have following format:
    [filename, threshold, actual class, predicted class, number of zero pixel, number of anomaly pixel ]
    '''
    with open(csv_path, 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #filewriter.writerow(['Filename', 'Threshold', 'actual class','predicted class','num normal Pixel','num anomaly Pixel'])
        #filewriter.writerow(['Derek', 'Software Developer'])
        filewriter.writerow(line_array)

def get_mp_classification(score,pthreshold):
    anomaly_score=score.max(axis=1).max()
    if pthreshold<anomaly_score:
        return 1,"anomaly", anomaly_score
    else:
        return 0,"good",anomaly_score

def get_classification(score_bw,area_threshold):
    '''
    returns 1,"anomaly" for anomaly
    and 0,"good" for good sample
    decision made by checking if 
    number of anomaly pixels of 
    the black and white score 
    is above area_threshold
    also returns number of pixel 
    that are considered as anomaly area
    '''
    anomaly_score=(score_bw>0).sum()
    if anomaly_score>area_threshold:
        return 1,"anomaly",anomaly_score
    else:
        return 0,"good",anomaly_score
    
def crop_torch_img(torch_img,croppingfactor,overlapfactor):
    '''
    takes torch tensor with one sample image (shape: [1,rgb,height,width])
    and a croppingfactor (eg. 4) that says what fraction of the image measurements the cropped ones have
    (eg 1/4 of length and width -> 16 cropped images)
    returns a list of cropped images in the following order:
    first column of image then second and so on
    '''
    
    height,width = torch_img.shape[2],torch_img.shape[3]
    #print(height,width)
    cropheight=int(height/croppingfactor)
    cropwidth=int(width/croppingfactor)
    overlapheight=int(cropheight*overlapfactor)
    overlapwidth=int(cropwidth*overlapfactor)
    w_rest=(width-cropwidth)%(cropwidth-overlapwidth)
    h_rest=(height-cropheight)%(cropheight-overlapheight)
    torch_img_rest = torch_img[:,:,h_rest:height,w_rest:width]
    #print((height-cropheight)/(cropheight-overlapheight))
    height,width=torch_img_rest.shape[2],torch_img_rest.shape[3] 
    list_cropped_torch_images=[]
    #print(h_rest,height,cropheight,overlapheight)
    #print((height-cropheight)/(cropheight-overlapheight))
    #print(w_rest,width,cropwidth,overlapwidth)
    stepsh=int((height-cropheight)/(cropheight-overlapheight)+1)
    stepsw=int((width-cropwidth)/(cropwidth-overlapwidth)+1)
    #sys.exit()
    for w in range(stepsw):
        for h in range(stepsh):
            torch_img_cropped=torch_img_rest[:,:,h*(cropheight-overlapheight):h*(cropheight-overlapheight)+cropheight,w*(cropwidth-overlapwidth):w*(cropwidth-overlapwidth)+cropwidth]
            #img_cropped=img_rest.crop((w*(width/factor),h*height/factor,(w+1)*width/factor,(h+1)*height/factor))
            list_cropped_torch_images.append(torch_img_cropped)
            
    return list_cropped_torch_images

def img_transposetorch2nparr(torchtensor):
    '''
    torch_tensor:   (1,rgb,height,width)
    np array image: (height,width,rgb)
    '''
    return torchtensor[0,:,:,:].transpose(1,2,0)

def concat_hm(torch_img,cropped_scores,croppingfactor,overlapfactor):
    '''
    concatenates heatmaps in following order:
    first column, second column and so on
    cropped_scoras: list of 2D arrays containing the heatmaps
    croppingfactor (eg. 4) that says what fraction of the image measurements the cropped ones have
    (eg 1/4 of length and width -> 16 cropped images))
    '''
    height,width=torch_img.shape[2],torch_img.shape[3]
    cropheight,cropwidth=cropped_scores[0].shape
    overlapheight=int(cropheight*overlapfactor)
    overlapwidth=int(cropwidth*overlapfactor)
    w_rest=(width-cropwidth)%(cropwidth-overlapwidth)
    h_rest=(height-cropheight)%(cropheight-overlapheight)

    height=height-h_rest
    width=width-w_rest
    #print(width,cropwidth,overlapfactor,overlapwidth,w_rest)
    #sys.exit()
    stepsh=int((height-cropheight)/(cropheight-overlapheight)+1)
    stepsw=int((width-cropwidth)/(cropwidth-overlapwidth)+1)
    score=np.zeros((cropheight*croppingfactor,cropwidth*croppingfactor))
    i=0
    img_innertop=overlapheight
    img_innerright=cropwidth-overlapwidth
    img_innerleft=overlapwidth
    img_innerbottom=cropheight-overlapheight
    for w in range(stepsw):
        for h in range(stepsh):
            #print(cropped_scores[i].shape)
            top=h*(cropheight-overlapheight)
            left=w*(cropwidth-overlapwidth)
            right=w*(cropwidth-overlapwidth)+cropwidth
            bottom=h*(cropheight-overlapheight)+cropheight
            innertop=h*(cropheight-overlapheight)+overlapheight
            innerleft=w*(cropwidth-overlapwidth)+overlapwidth
            innerright=(w+1)*(cropwidth-overlapwidth)
            innerbottom=(h+1)*(cropheight-overlapheight)
            overlaymethod="max"
            if overlaymethod=="max":
                score[top:bottom,left:right]=np.maximum(cropped_scores[i],score[top:bottom,left:right])
            elif overlaymethod=="min":
                score[top:bottom,left:right]=np.minimum(cropped_scores[i],score[top:bottom,left:right])
            elif overlaymethod=="average":

                if w==0 and h==0:
                    score[top:innerbottom,left:innerright]=cropped_scores[i][0:img_innerbottom,0:img_innerright]
                    score[innerbottom:bottom,left:innerright]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerright]/2)+score[innerbottom:bottom,left:innerright]
                    score[top:innerbottom,innerright:right]=(cropped_scores[i][0:img_innerbottom,img_innerright:cropwidth]/2)+score[top:innerbottom,innerright:right]
                    score[innerbottom:bottom,innerright:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerright:cropwidth]/4)+score[innerbottom:bottom,innerright:right]
                elif w==0 and 0<h<stepsh-1:
                    score[top:innertop,left:innerright]=(cropped_scores[i][0:img_innertop,0:img_innerright]/2)+score[top:innertop,left:innerright]               
                    score[top:innertop,innerright:right]=(cropped_scores[i][0:img_innertop,img_innerright:cropwidth]/4)+score[top:innertop,innerright:right]
                    score[innertop:innerbottom,left:innerright]=cropped_scores[i][img_innertop:img_innerbottom,0:img_innerright]
                    score[innertop:innerbottom,innerright:right]=(cropped_scores[i][img_innertop:img_innerbottom,img_innerright:cropwidth]/2)+score[innertop:innerbottom,innerright:right]
                    score[innerbottom:bottom,left:innerright]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerright]/2)+score[innerbottom:bottom,left:innerright]
                    score[innerbottom:bottom,innerright:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerright:cropwidth]/4)+score[innerbottom:bottom,innerright:right]                    
                elif w==0 and h==stepsh-1:
                    score[top:innertop,left:innerright]=(cropped_scores[i][0:img_innertop,0:img_innerright]/2)+score[top:innertop,left:innerright]
                    score[top:innertop,innerright:right]=(cropped_scores[i][0:img_innertop,img_innerright:cropwidth]/4)+score[top:innertop,innerright:right]
                    score[innertop:bottom,left:innerright]=cropped_scores[i][img_innertop:cropheight,0:img_innerright]
                    score[innertop:bottom,innerright:right]=(cropped_scores[i][img_innertop:cropheight,img_innerright:cropwidth]/2)+score[innertop:bottom,innerright:right]
                elif 0<w<stepsw-1 and h==0:
                    score[top:innerbottom,left:innerleft]=(cropped_scores[i][0:img_innerbottom,0:img_innerleft]/2)+score[top:innerbottom,left:innerleft]
                    score[top:innerbottom,innerleft:innerright]=cropped_scores[i][0:img_innerbottom,img_innerleft:img_innerright]
                    score[top:innerbottom,innerright:right]=(cropped_scores[i][0:img_innerbottom,img_innerright:cropwidth]/2)+score[top:innerbottom,innerright:right]
                    score[innerbottom:bottom,left:innerleft]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerleft]/4)+score[innerbottom:bottom,left:innerleft]
                    score[innerbottom:bottom,innerleft:innerright]=(cropped_scores[i][img_innerbottom:cropheight,img_innerleft:img_innerright]/4)+score[innerbottom:bottom,innerleft:innerright]
                    score[innerbottom:bottom,innerright:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerright:cropwidth]/4)+score[innerbottom:bottom,innerright:right]  
                elif 0<w<stepsw-1 and 0<h<stepsh-1:
                    score[top:innertop,left:innerleft]=(cropped_scores[i][0:img_innertop,0:img_innerleft]/4)+ score[top:innertop,left:innerleft]
                    score[top:innertop,innerleft:innerright]=(cropped_scores[i][0:img_innertop,img_innerleft:img_innerright]/2)+ score[top:innertop,innerleft:innerright]
                    score[top:innertop,innerright:right]=(cropped_scores[i][0:img_innertop,img_innerright:cropwidth]/4)+score[top:innertop,innerright:right]
                    score[innertop:innerbottom,left:innerleft]=(cropped_scores[i][img_innertop:img_innerbottom,0:img_innerleft]/2)+score[innertop:innerbottom,left:innerleft]
                    score[innertop:innerbottom,innerleft:innerright]=cropped_scores[i][img_innertop:img_innerbottom,img_innerleft:img_innerright]
                    score[innertop:innerbottom,innerright:right]=(cropped_scores[i][img_innertop:img_innerbottom,img_innerright:cropwidth]/2)+score[innertop:innerbottom,innerright:right]
                    score[innerbottom:bottom,left:innerleft]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerleft]/4)+score[innerbottom:bottom,left:innerleft]
                    score[innerbottom:bottom,innerleft:innerright]=(cropped_scores[i][img_innerbottom:cropheight,img_innerleft:img_innerright]/2)+score[innerbottom:bottom,innerleft:innerright]
                    score[innerbottom:bottom,innerright:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerright:cropwidth]/4)+score[innerbottom:bottom,innerright:right]
                elif 0<w<stepsw-1 and h==stepsh-1:
                    score[top:innertop,left:innerleft]=(cropped_scores[i][0:img_innertop,0:img_innerleft]/4)+ score[top:innertop,left:innerleft]
                    score[top:innertop,innerleft:innerright]=(cropped_scores[i][0:img_innertop,img_innerleft:img_innerright]/2)+ score[top:innertop,innerleft:innerright]
                    score[top:innertop,innerright:right]=(cropped_scores[i][0:img_innertop,img_innerright:cropwidth]/4)+score[top:innertop,innerright:right]
                    score[innertop:bottom,left:innerleft]=(cropped_scores[i][img_innertop:cropheight,0:img_innerleft]/2)+score[innertop:bottom,left:innerleft]
                    score[innertop:bottom,innerleft:innerright]=cropped_scores[i][img_innertop:cropheight,img_innerleft:img_innerright]
                    score[innertop:bottom,innerright:right]=(cropped_scores[i][img_innertop:cropheight,img_innerright:cropwidth]/2)+score[innertop:bottom,innerright:right]
                elif w==stepsw-1 and h==0:
                    score[top:innerbottom,left:innerleft]=(cropped_scores[i][0:img_innerbottom,0:img_innerleft]/2)+score[top:innerbottom,left:innerleft]
                    score[innerbottom:bottom,left:innerleft]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerleft]/4)+score[innerbottom:bottom,left:innerleft]
                    score[top:innerbottom,innerleft:right]=cropped_scores[i][0:img_innerbottom,img_innerleft:cropwidth]
                    score[innerbottom:bottom,innerleft:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerleft:cropwidth]/2)+score[innerbottom:bottom,innerleft:right]
                elif w==stepsw-1 and 0<h<stepsh-1:
                    score[top:innertop,left:innerleft]=(cropped_scores[i][0:img_innertop,0:img_innerleft]/4)+ score[top:innertop,left:innerleft]
                    score[top:innertop,innerleft:right]=(cropped_scores[i][0:img_innertop,img_innerleft:cropwidth]/2)+score[top:innertop,innerleft:right]
                    score[innertop:innerbottom,left:innerleft]=(cropped_scores[i][img_innertop:img_innerbottom,0:img_innerleft]/2)+score[innertop:innerbottom,left:innerleft]
                    score[innertop:innerbottom,innerleft:right]=cropped_scores[i][img_innertop:img_innerbottom,img_innerleft:cropwidth]
                    score[innerbottom:bottom,left:innerleft]=(cropped_scores[i][img_innerbottom:cropheight,0:img_innerleft]/4)+score[innerbottom:bottom,left:innerleft]
                    score[innerbottom:bottom,innerleft:right]=(cropped_scores[i][img_innerbottom:cropheight,img_innerleft:cropwidth]/2)+score[innerbottom:bottom,innerleft:right]
                elif w==stepsw-1 and h==stepsh-1:
                    score[top:innertop,left:innerleft]=(cropped_scores[i][0:img_innertop,0:img_innerleft]/4)+ score[top:innertop,left:innerleft]
                    score[top:innertop,innerleft:right]=(cropped_scores[i][0:img_innertop,img_innerleft:cropwidth]/2)+score[top:innertop,innerleft:right]
                    score[innertop:bottom,left:innerleft]=(cropped_scores[i][img_innertop:cropheight,0:img_innerleft]/2)+score[innertop:bottom,left:innerleft]
                    score[innertop:bottom,innerleft:right]=cropped_scores[i][img_innertop:cropheight,img_innerleft:cropwidth]
            i+=1
    return gaussian_filter(score, sigma=4)


def th_method_AUROC(prediction_actual_list):
    '''
    input: list with entrys in format:[prediction,actual]
    with 0=good, 1=anomaly
    output ROC value
    '''
    rocdata=[]
    tp=0
    fp=0
    for i in range(len(prediction_actual_list)):
        if prediction_actual_list[i][0]==1 and prediction_actual_list[i][1]==1:
            tp+=1
        elif prediction_actual_list[i][0]==1 and prediction_actual_list[i][1]==0:
            fp+=1
        rocdata.append([tp,fp])
    


    return rocdata

def get_hm_dir(hm_dir_basis,sorting,prediction,actual):
    if not os.path.isdir(hm_dir_basis):
            os.mkdir(hm_dir_basis)
    
    if sorting:
        
        
        hm_dir=hm_dir_basis+("actual anomaly/" if actual==1 else "actual good/")
        if not os.path.isdir(hm_dir):
            os.mkdir(hm_dir)
        if prediction<actual: #pred good but actual anomaly
            hm_dir=hm_dir+"false_negative/"
            
        elif prediction>actual:
            hm_dir=hm_dir+"false_positive/"
    else:
        hm_dir=hm_dir_basis
    if not os.path.isdir(hm_dir):
            os.mkdir(hm_dir)
    return hm_dir

def save_csv_hm(sample,score,hm_dir_basis,hm_sorting,csv_path,th,area_th,blendingfactor,pmaxthreshold):
    image=sample['imageBase']
    label=sample["has_anomaly"]
    f, axarr = plt.subplots(3,2)
    img=img_transposetorch2nparr(image.cpu().numpy()) #numpy image: (height,width,rgb)
    img=(cv2.normalize(img,None,0,1,cv2.NORM_MINMAX)*255).astype(np.uint8)
    axarr[0][0].imshow(img)
    normalized_hm = cv2.normalize(score, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply a colormap to the heatmap
    colormap = plt.get_cmap('viridis')
    normalized_hm_colored = colormap(normalized_hm)
    
    # Convert the heatmap to RGB
    normalized_hm_colored8 = (normalized_hm_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(normalized_hm_colored8,blendingfactor,img,1-blendingfactor,0)#img_transposetorch2nparr(image.cpu().numpy())*0.7#+heatmap_colored*0.3
    im_hm01=axarr[0][1].imshow(overlay)
    im_hm10=axarr[1][0].imshow(normalized_hm_colored8)
    f.colorbar(im_hm10,location="left")
    
    im_hm20=axarr[2][0].imshow(score, interpolation='nearest', cmap='viridis',vmin=0.0001,vmax=0.0002)#,vmin=0,vmax=0.0002)
    f.colorbar(im_hm20,location="left")
    
    score_bw=th_img(score,th)
    score_bw_norm=th_img(normalized_hm,th)
    
    im_hm11=axarr[2][1].imshow(score_bw) #, interpolation='nearest', cmap='viridis',vmin=0.0001,vmax=0.0002)
    im_hm12=axarr[1][1].imshow(score_bw_norm, interpolation='nearest', cmap='viridis',vmin=0,vmax=1)
    #f.colorbar(im_hm2,location="right")
    hmprediction,hmpred_str,hmnum_anomalypixel=get_classification(score_bw,area_th)
    prediction,pred_str,num_anomalypixel=get_mp_classification(score,pmaxthreshold)
    actual_str=("anomaly" if label.cpu().numpy()[0][0]==1 else "good")
    hm_dir=get_hm_dir(hm_dir_basis,hm_sorting,prediction,label.cpu().numpy()[0][0])
    plt.savefig(hm_dir+str(sample["file_name"])+"_predicted-"+pred_str+"__actual-"+actual_str+"__numanomalypixel-" +str(num_anomalypixel)+'.png')
    csv_arr=[str(th),str(label.cpu().numpy()[0][0]),str(prediction),str((score_bw == 0).sum()),str((score_bw > 0).sum())]
    #threshold bw; actual class; prediction; num zero pixel; num not zero pixel
    write_in_csv(csv_path,csv_arr)
    plt.close(f)
    return num_anomalypixel,hmnum_anomalypixel

def generate_result_path(trainer):
    test_timestamp=str(datetime.now().hour)+"_"+str(datetime.now().minute)
    if not os.path.isdir(trainer.save_path):
        os.mkdir(trainer.save_path)
    if not os.path.isdir(trainer.save_path+'/csv/'):
        os.mkdir(trainer.save_path+'/csv/')
    csv_path=trainer.save_path+'/csv/'+trainer.param_str+"_"+test_timestamp+".csv"
    hm_dir_basis=trainer.save_path+'/heatmaps/'
    if not os.path.isdir(hm_dir_basis):
        os.mkdir(hm_dir_basis)
    hm_dir_basis=hm_dir_basis+trainer.param_str+"/"
    #print("generate_result_path-functions.py",hm_dir_basis)
    if not os.path.isdir(hm_dir_basis):
        os.mkdir(hm_dir_basis)
    hm_dir_basis=hm_dir_basis+test_timestamp+"/"
    if not os.path.isdir(hm_dir_basis):
        os.mkdir(hm_dir_basis)

    return hm_dir_basis,csv_path,test_timestamp
        
def augmented_scores(trainer, img_input): # -> List[np.array]:
        score = trainer.score(img_input)
        score_list = [score]

        if trainer.rot_90:
            rotated_90 = TF.rotate(img_input, -90)
            rotated_90_score = trainer.score(rotated_90)
            rotated_90_score = np.rot90(rotated_90_score)
            score_list.append(rotated_90_score)
        if trainer.rot_180:
            rotated_180 = TF.rotate(img_input, -180)
            rotated_180_score = trainer.score(rotated_180)
            rotated_180_score = np.rot90(rotated_180_score, k=2)
            score_list.append(rotated_180_score)
        if trainer.rot_270:
            rotated_270 = TF.rotate(img_input, -270)
            rotated_270_score = trainer.score(rotated_270)
            rotated_270_score = np.rot90(rotated_270_score, k=3)
            score_list.append(rotated_270_score)
        if trainer.h_flip:
            horizontal_flip = torch.flip(img_input, dims=[3])
            horizontal_flip_score = trainer.score(horizontal_flip)
            horizontal_flip_score = np.fliplr(horizontal_flip_score)
            score_list.append(horizontal_flip_score)
        if trainer.h_flip_rot_90:
            flipped_rotated_90 = TF.rotate(torch.flip(img_input, dims=[3]), -90)
            flipped_rotated_90_score = trainer.score(flipped_rotated_90)
            flipped_rotated_90_score = np.fliplr(np.rot90(flipped_rotated_90_score))
            score_list.append(flipped_rotated_90_score)
        if trainer.h_flip_rot_180:
            flipped_rotated_180 = TF.rotate(torch.flip(img_input, dims=[3]), -180)
            flipped_rotated_180_score = trainer.score(flipped_rotated_180)
            flipped_rotated_180_score = np.fliplr(np.rot90(flipped_rotated_180_score, k=2))
            score_list.append(flipped_rotated_180_score)
        if trainer.h_flip_rot_270:
            flipped_rotated_270 = TF.rotate(torch.flip(img_input, dims=[3]), -270)
            flipped_rotated_270_score = trainer.score(flipped_rotated_270)
            flipped_rotated_270_score = np.fliplr(np.rot90(flipped_rotated_270_score, k=3))
            score_list.append(flipped_rotated_270_score)
        return score_list

def __mean_scores(self, score_list):
    res = np.mean(score_list, axis=0)

    return res

def save_log_csv(trainer,confusionmatrix,th,test_timestamp,img_roc_auc):
    if not os.path.isdir(trainer.save_path+'/logs/'):
        os.mkdir(trainer.save_path+'/logs/')
    csv_path=trainer.save_path+'/logs/log.csv'
    tp=str(confusionmatrix[0][0].item())
    fp=str(confusionmatrix[1][0].item())
    tn=str(confusionmatrix[1][1].item())
    fn=str(confusionmatrix[0][1].item())
    if trainer.myworkswitch:
        my_work_label=trainer.myworklabel
    else: 
        my_work_label="OG architecture"
    

    log_arr=[trainer.param_str,my_work_label,test_timestamp,(trainer.traintime if trainer.phase=="train" else "test"),img_roc_auc,tp,fp,tn,fn,str(th)]
    write_in_csv(csv_path,log_arr)

#                   ((true positive, false negative)
        #            (false positive, true negative))