import torch
import sys
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import cv2
import csv
from PIL import Image
import numpy as np
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
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
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
    
def crop_torch_img(torch_img,croppingfactor):
    '''
    takes torch tensor with one sample image (shape: [1,rgb,height,width])
    and a croppingfactor (eg. 4) that says what fraction of the image measurements the cropped ones have
    (eg 1/4 of length and width -> 16 cropped images)
    returns a list of cropped images in the following order:
    first column of image then second and so on
    '''
    height,width = torch_img.shape[2],torch_img.shape[3] 
    w_rest=width%croppingfactor
    h_rest=height%croppingfactor
    torch_img_rest = torch_img[:,:,h_rest:height,w_rest:width]
    height,width=torch_img_rest.shape[2],torch_img_rest.shape[3] 
    list_cropped_torch_images=[]
    for w in range(croppingfactor):
        for h in range(croppingfactor):
            torch_img_cropped=torch_img_rest[:,:,int(h*(height/croppingfactor)):int((h+1)*height/croppingfactor),int(w*width/croppingfactor):int((w+1)*width/croppingfactor)]
            #img_cropped=img_rest.crop((w*(width/factor),h*height/factor,(w+1)*width/factor,(h+1)*height/factor))
            list_cropped_torch_images.append(torch_img_cropped)
            
    return list_cropped_torch_images


def img_transposetorch2nparr(torchtensor):
    '''
    torch_tensor:   (1,rgb,height,width)
    np array image: (height,width,rgb)
    '''
    return torchtensor[0,:,:,:].transpose(1,2,0)



def concat_hm(cropped_scores,croppingfactor):
    '''
    concatenates heatmaps in following order:
    first column, second column and so on
    cropped_scoras: list of 2D arrays containing the heatmaps
    croppingfactor (eg. 4) that says what fraction of the image measurements the cropped ones have
    (eg 1/4 of length and width -> 16 cropped images))
    '''
    heigth,width=cropped_scores[0].shape
    score=np.zeros((heigth*croppingfactor,width*croppingfactor))
    i=0
    for w in range(croppingfactor):
        for h in range(croppingfactor):
            #print(cropped_scores[i].shape)
            score[h*heigth:(h+1)*heigth,w*width:(w+1)*width]=cropped_scores[i]
            i+=1
    return score

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


