import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve,RocCurveDisplay
from models.teacher import teacherTimm
from models.StudentTeacher.student  import studentTimm
from datasets.mvtec import MVTecDataset
from models.EfficientAD.efficientAD import loadPdnTeacher
from models.EfficientAD.common import get_pdn_medium,get_pdn_small
from models.ReverseDistillation.rd import loadBottleNeckRD, loadStudentRD
from models.DBFAD.reverseResidual import reverse_student18
from torcheval.metrics import BinaryConfusionMatrix
from torchvision import datasets, transforms
import sys
from PIL import Image
import numpy as np
from utils.functions import write_in_csv

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("runs")



def getParams(trainer,data,device):
    trainer.device = device
    trainer.validation_ratio = 0.2
    trainer.data_path = data['data_path']
    trainer.debugmode=data['debugparams']
    trainer.phase=data['phase']
    
    trainer.obj = data['obj']
    trainer.num_epochs = data['TrainingData']['epochs']
    trainer.img_resize_h = data['TrainingData']['img_size_h']
    trainer.img_resize_w = data['TrainingData']['img_size_w']
    trainer.img_cropsize = data['TrainingData']['crop_size'] #for centercrop mvtec
    trainer.lr = data['TrainingData']['lr']
    trainer.batch_size = data['TrainingData']['batch_size'] 
    trainer.myworkswitch=data['myworkswitch'] 
    trainer.myworklabel=data['myworklabel']
    trainer.augmentation=data['augmentation']
    if data['myworkswitch']:
        trainer.save_path = data['save_path_modified_architecture']+"/"+data['myworklabel']
    else: 
        trainer.augmentation=False
        trainer.save_path = data['save_path']
    trainer.handmade=data["handmade"]
    trainer.write=data['write']
    trainer.hm_sorting=data['hm_sorting']
    trainer.blendfactor=data['blendfactor']
    trainer.model_dir = trainer.save_path+ "/models" + "/" + trainer.obj  
    trainer.img_dir = trainer.save_path+ "/imgs" + "/" + trainer.obj 
    trainer.modelName = data['backbone']
    trainer.outIndices = data['out_indice']
    trainer.distillType=data['distillType']
    trainer.norm = data['TrainingData']['norm']
    trainer.threshold=data['threshold']
    trainer.param_str=str(data['obj'])+"_"+str("NOTcropped" if data['cropping'] else "cropped")+"_"+str(data['TrainingData']['epochs'])+"_"+str(data['TrainingData']['batch_size'])+"_"+str(data['TrainingData']['lr'])
    trainer.cropping=data['cropping'] #my own cropping for hd image downsizing
    trainer.croppingfactor=data['croppingfactor']
    trainer.overlapfactor=data['overlapfactor']
    trainer.test_img_resize_h = data['TestData']['img_size_h']
    trainer.test_img_resize_w = data['TestData']['img_size_w']
    trainer.test_img_cropsize = data['TestData']['crop_size']
    trainer.rot_90=data['AugmentScores']['rot_90']
    trainer.rot_180=data['AugmentScores']['rot_180']
    trainer.rot_270=data['AugmentScores']['rot_270']
    trainer.h_flip=data['AugmentScores']['h_flip']
    trainer.h_flip_rot_90=data['AugmentScores']['h_flip_rot_90']
    trainer.h_flip_rot_180=data['AugmentScores']['h_flip_rot_180']
    trainer.h_flip_rot_270=data['AugmentScores']['h_flip_rot_270']

    if data['debugparams']:
        trainer.obj = "fabric_not_cropped_DEBUGGING"
        trainer.num_epochs = 1
        trainer.save_path="./results_Debugging"
    
def loadWeights(model,model_dir,alias):
    #print("loadWeights")
    try:
        checkpoint = torch.load(os.path.join(model_dir, alias))
    except:
        raise Exception("Check saved model path.")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    #print("load weights",model.parameters()) 
    for param in model.parameters():
        param.requires_grad = False
    return model

def loadTeacher(trainer):
    #print("loadTeacher")
    if (trainer.distillType=="st"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName,out_indices=trainer.outIndices).to(trainer.device)
    elif (trainer.distillType=="ead"):
        loadPdnTeacher(trainer)
    elif (trainer.distillType=="rd"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName,out_indices=[1,2,3]).to(trainer.device)
        loadBottleNeckRD(trainer)
    elif (trainer.distillType=="dbfad"):
        trainer.teacher=teacherTimm(backbone_name="resnet34",out_indices=[0,1,2,3]).to(trainer.device)
    elif (trainer.distillType=="mixed"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName[0],out_indices=trainer.outIndices[0]).to(trainer.device)
        trainer.teacher2=teacherTimm(backbone_name=trainer.modelName[1],out_indices=trainer.outIndices[1]).to(trainer.device)
        trainer.teacher2.eval()
        for param in trainer.teacher2.parameters():
            param.requires_grad = False
    else:
        raise Exception("Invalid distillation type :  Choices are ['st', 'ead','rd', 'dbfad']")
    
    # load bottleneck rd
    trainer.teacher.eval()
    for param in trainer.teacher.parameters():
        param.requires_grad = False

def loadModels(trainer):
    #print("loadModels")
    if (trainer.distillType=="st"):
        loadTeacher(trainer)
        trainer.student=studentTimm(backbone_name=trainer.modelName,out_indices=trainer.outIndices).to(trainer.device)
    if (trainer.distillType=="ead"):
        loadTeacher(trainer)
        if trainer.modelName=="small":
            trainer.student = get_pdn_small().to(trainer.device) # 768 if autoencoder
        if trainer.modelName=="medium" : 
            trainer.student = get_pdn_medium().to(trainer.device) # 768 if autoencoder
    if (trainer.distillType=="rd"):
        loadTeacher(trainer)
        loadStudentRD(trainer)
    if (trainer.distillType=="dbfad"):
        loadTeacher(trainer)
        trainer.student=reverse_student18().to(trainer.device)
    if (trainer.distillType=="mixed"):
        loadTeacher(trainer)
        trainer.student=studentTimm(backbone_name=trainer.modelName[0],out_indices=trainer.outIndices[0]).to(trainer.device)
        trainer.student2=studentTimm(backbone_name=trainer.modelName[1],out_indices=trainer.outIndices[1]).to(trainer.device)
    
def loadDataset(trainer):
    '''
    cropping set to False, croppingfactor to None
    '''
    #print("loadDataset")
    kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
    train_dataset = MVTecDataset(root_dir=trainer.data_path+"/"+trainer.obj+"/train/good",
        resize_shape=[trainer.img_resize_h,trainer.img_resize_w],
        crop_size=[trainer.img_cropsize,trainer.img_cropsize],
        phase='train',
        croppingfactor=None,
        cropping=False,
        augmentation=trainer.augmentation
    )
    img_nums = len(train_dataset)
    valid_num = int(img_nums * trainer.validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_num, valid_num]
    )

                                                                  #kwargs=num_workers etc dependent on gpu available or not
    trainer.train_loader=torch.utils.data.DataLoader(train_data, batch_size=trainer.batch_size, shuffle=True, **kwargs) 
    
    trainer.val_loader=torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)
    #return train_data,val_data

# def get_random_transforms():
#     augmentative_transforms = []
#     if c.transf_rotations:
#         augmentative_transforms += [transforms.RandomRotation(180)]
#     if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
#         augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
#                                                            saturation=c.transf_saturation)]

#     tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
#                                                                        transforms.Normalize(c.norm_mean, c.norm_std)]

#     transform_train = transforms.Compose(tfs)
#     return transform_train


# def get_fixed_transforms(degrees):
#     cust_rot = lambda x: rotate(x, degrees, False, False, None)
#     augmentative_transforms = [cust_rot]
#     if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
#         augmentative_transforms += [
#             transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
#                                    saturation=c.transf_saturation)]
#     tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
#                                                                        transforms.Normalize(c.norm_mean,
#                                                                                             c.norm_std)]
#     return transforms.Compose(tfs)

    
def infer(trainer, img):
    #print("infer")
    #img.type(torch.float64)
    trainer.val_loader
 
    if (trainer.distillType=="st" ):
        features_t = trainer.teacher(img)
        features_s=trainer.student(img)
    if (trainer.distillType=="ead"):
        features_t = [trainer.teacher(img)]
        features_s=[trainer.student(img)]
    if (trainer.distillType=="rd"):
        features_t = trainer.teacher(img)
        embed=trainer.bn(features_t)
        features_s=trainer.student(embed)
    if (trainer.distillType=="dbfad"):
        features_t = trainer.teacher(img)
        features_t = [F.max_pool2d(features_t[0],kernel_size=3,stride=2,padding=1),features_t[1],features_t[2],features_t[3]]
        features_s=trainer.student(features_t)
        #writer.add_graph(trainer.student,features_t)
    if (trainer.distillType=="mixed"):
        features_t = trainer.teacher(img)
        features_t2 = trainer.teacher2(img)
        features_s=trainer.student(img)
        features_s2=trainer.student2(img) 
        features_s=list(features_s)+list(features_s2)
        features_t=list(features_t)+list(features_t2)
    return features_s,features_t

def computeAUROC(trainer,scores,gt_list,obj,name="base"):
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    img_scores_tmp = scores.reshape(scores.shape[0], -1)



    img_scores=img_scores_tmp.max(axis=1)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(obj + " image"+str(name)+" ROCAUC: %.3f" % (img_roc_auc))

    csvpath='/home/christianjaspert/masterthesis/DistillationAD/areaTHstatistics.csv'
    for th in range(10):
        img_scores_area=[]
        for i in range(img_scores_tmp.shape[0]):
            numanomalpixel=np.sum(img_scores_tmp[i]>th*0.02+0.1)
            img_scores_area.append(numanomalpixel)
            csvarray=[numanomalpixel,gt_list[i][0],th*0.02+0.1]
            write_in_csv(csvpath,csvarray)
        img_scores_area=np.asarray(img_scores_area)/len(img_scores_tmp[0])
        img_roc_auc_area = roc_auc_score(gt_list, img_scores_area)
    print(obj + " image"+str(name)+" AREA ROCAUC: %.3f" % (img_roc_auc_area))
    



    _1,_2,ths=computeROCcurve(gt_list,img_scores)
    roc_curve=RocCurveDisplay.from_predictions(np.array(gt_list[:,0]),np.array(img_scores)).figure_
    if trainer.write:
        writer.add_pr_curve("Precision Recall Curve "+obj,np.array(gt_list[:,0]),np.array(img_scores),None,100)
        writer.add_figure("ROC curve "+obj,roc_curve)

    for i in range(10):
        metric=BinaryConfusionMatrix(threshold=0.1*i)
        metric.update(torch.tensor(img_scores),torch.tensor(gt_list[:,0]))
        if i==0:
            optmatrix=metric.compute()
            optth=i
        else:
            if optmatrix[0][0]+optmatrix[1][1]<metric.compute()[0][0]+metric.compute()[1][1]:
                optmatrix=metric.compute()
                optth=i
        #print(metric.compute())
    print("Optimal Matrix for th %.3f:" %(optth))
    print(optmatrix)
    return img_roc_auc,img_scores,optmatrix,optth

def cal_importance(ft, fs,norm):

    fs_norm = F.normalize(fs, p=2) if norm else fs
    ft_norm = F.normalize(ft, p=2) if norm else ft

    f_loss = 0.5 * (ft_norm - fs_norm) ** 2

    sumOverAxes=torch.sum(f_loss,dim=[1,2])
    sortedIndex=torch.argsort(sumOverAxes,descending=True)
    ft_norm = ft_norm[sortedIndex]
    fs_norm = fs_norm[sortedIndex]
    
    return ft_norm,fs_norm

def computeROCcurve(y_true,y_score):
    '''
    return:
    fpr, tpr, thresholds
    '''
    fpr, tpr, thresholds=roc_curve(np.array(y_true),np.array(y_score))
    return fpr, tpr, thresholds
