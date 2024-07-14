import os
from datetime import datetime
import sys
from PIL import Image
from matplotlib import pyplot as plt
import time
import numpy as np
import torch
from torcheval.metrics import BinaryConfusionMatrix
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from utils.util import  AverageMeter,readYamlConfig,set_seed
from utils.functions import (
    cal_loss,
    cal_anomaly_maps,
    th_img,
    write_in_csv,
    get_classification,
    crop_torch_img,
    img_transposetorch2nparr,
    concat_hm,
    
)
from utilsTraining import getParams,loadWeights,loadModels,loadDataset,infer,computeAUROC,computeROCcurve
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("runs")
torch.set_default_dtype(torch.float64)




####################################################
#DONT DELETE!!!!!!!!!!!!!!
#the commented code below is to show in wich file a print was used!
####################################################

# import sys
# import inspect
# import collections
# _stdout = sys.stdout

# Record = collections.namedtuple(
#     'Record',
#     'frame filename line_number function_name lines index')
# class MyStream(object):
#     def __init__(self, target):
#         self.target = target
#     def write(self, text):
#         if text.strip():
#             record = Record(*inspect.getouterframes(inspect.currentframe())[1])        
#             self.target.write(
#                 '{f} {n}: '.format(f = record.filename, n = record.line_number))
#         self.target.write(text)

# sys.stdout = MyStream(sys.stdout)

# def foo():
#     print('Hi')

# foo()


class NetTrainer:          
    def __init__(self, data,device,stats,epochs):  
        
        getParams(self,data,device)
        os.makedirs(self.model_dir, exist_ok=True)
        if stats:
            self.num_epochs=epochs
        # You can set seed for reproducibility
        set_seed(42)
        
        loadModels(self)
        loadDataset(self)
        #print(type(train_data))
        #print("ohne np",train_dataset.__getitem__(0).get("imageBase").shape)              
        #print("mit np",np.asarray(train_dataset.__getitem__(0).get("imageBase")).shape)
        
        if self.distillType=="rd":
            self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.bn.parameters()), lr=self.lr, betas=(0.5, 0.999)) 
        elif self.distillType=="mixed" :
            self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.student2.parameters()), lr=self.lr, betas=(0.5, 0.999))
        else:
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lr*10,epochs=self.num_epochs,steps_per_epoch=len(self.train_loader))
        

    def train(self):
        print("training " + self.obj)
        self.student.train() 
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        losslist=[]

        for _ in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            loss_epoch=0
            for sample in self.train_loader:
                image = sample['imageBase'].to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):

                    features_s,features_t  = infer(self,image) 
                    loss=cal_loss(features_s, features_t,trainer.norm)
                    loss_epoch=loss_epoch+loss/len(self.train_loader)
                    losses.update(loss.sum().item(), image.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()
                #0print(loss_epoch.item())
            writer.add_scalar('training loss '+str(self.obj),loss_epoch,_)
               
            
            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
            
            writer.add_scalars('losses lr: '+str(self.obj)+": "+str(self.lr),{'training loss':loss_epoch.item(),'validation loss':val_loss},_)
            losslist.append([loss_epoch.item(),val_loss])
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            print("test")
        epoch_bar.close()
        print("Training end.")
        return losslist      

    def val(self, epoch_bar):
        self.student.eval()
        losses = AverageMeter()
        for sample in self.val_loader: 
            image = sample['imageBase'].to(self.device)
            with torch.set_grad_enabled(False):
                
                features_s,features_t  = infer(self,image)  

                loss=cal_loss(features_s, features_t,trainer.norm)
                                
                losses.update(loss.item(), image.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        if self.distillType=="rd":
            state = {"model": self.bn.state_dict()}
            torch.save(state, os.path.join(self.model_dir, "bn.pth"))
        if self.distillType=="mixed":
            state = {"model": self.student2.state_dict()}
            torch.save(state, os.path.join(self.model_dir, "student2.pth"))

    
    @torch.no_grad()

    ######################################################################################################################################
    def test(self):

        self.student=loadWeights(self.student,self.model_dir,"student.pth")
        if self.distillType=="rd":
            self.bn=loadWeights(self.bn,self.model_dir,"bn.pth")
        if self.distillType=="mixed":
            self.student2=loadWeights(self.student2,self.model_dir,"student2.pth")
        
        
        kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {} )
        
        test_dataset = MVTecDataset(
            root_dir=self.data_path+"/"+self.obj+"/test/",
            resize_shape=[self.img_resize_h,self.img_resize_w],
            crop_size=[self.img_cropsize,self.img_cropsize],
            phase='test',
            croppingfactor=trainer.croppingfactor,
            cropping=trainer.cropping
        )
        print()
        tag="idx: "+str(test_dataset.__getitem__(0).get("idx"))+"  has anomaly: "+str(test_dataset.__getitem__(0).get("has_anomaly"))+"  filename: "+str(test_dataset.__getitem__(0).get("file_name"))
        writer.add_image(tag,test_dataset.__getitem__(0).get("imageBase"))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        
        
        
        scores = []
        gt_list = []
        class_score=0
        progressBar = tqdm(test_loader)
        th=0.00018
        area_th=100
        test_timestamp=str(datetime.now().hour)+"_"+str(datetime.now().minute)
        hm_dir=trainer.save_path+'/heatmaps/'+trainer.param_str+"/"+test_timestamp+"/"
        if not os.path.isdir(hm_dir):
            os.mkdir(hm_dir)
        csv_path=trainer.save_path+'/csv/'+trainer.param_str+".csv"
        ctr=0
        y_true=[]
        y_score=[]
        
        for sample in test_loader:
            
            #print("test")
            label=sample['has_anomaly']
            y_true.append(label.cpu().numpy()[0][0])
            image = sample['imageBase'].to(self.device)
            gt_list.extend(label.cpu().numpy())
            concat_prediction=0
            
            with torch.set_grad_enabled(False):
                if trainer.cropping:
                    print("cropping")
                    cropped_scores=[]
                    for cropped_img in crop_torch_img(image,trainer.croppingfactor):
                        features_s, features_t = infer(self,cropped_img)
                        score=cal_anomaly_maps(features_s,features_t,self.img_cropsize,trainer.norm)
                        cropped_scores.append(score)


                    score=concat_hm(cropped_scores,trainer.croppingfactor)


                else:
                    print("not cropping")
                    features_s, features_t = infer(self,image)  
                    score =cal_anomaly_maps(features_s,features_t,self.img_cropsize,trainer.norm) 
                
                f, axarr = plt.subplots(1,3)
                im_fab=axarr[0].imshow(img_transposetorch2nparr(image.cpu().numpy()))#.astype('uint8'))
                #numpy image: (height,width,rgb)
                im_hm=axarr[1].imshow(score, interpolation='nearest', cmap='viridis',vmin=0,vmax=0.0002)#,vmin=0,vmax=0.0002)
                f.colorbar(im_hm)
                score_bw=th_img(score,th)
                im_hm2=axarr[2].imshow(score_bw, interpolation='nearest', cmap='viridis',vmin=0,vmax=0.0002)
                f.colorbar(im_hm2)
                prediction,pred_str,num_anomalypixel=get_classification(score_bw,area_th)

                plt.savefig(hm_dir+str(ctr)+"_predicted-"+pred_str+"__actual-"+("anomaly" if label.cpu().numpy()[0][0]==1 else "good")+"__numanomalypixel-" +str(num_anomalypixel)+'.png')
                csv_arr=[str(class_score),str(th),str(label.cpu().numpy()[0][0]),str(prediction),str((score_bw == 0).sum()),str((score_bw > 0).sum())]
                write_in_csv(csv_path,csv_arr)
                plt.close(f)
            if label.cpu().numpy()[0][0]==concat_prediction:
                class_score+=1

                #progressBar.update()

            ctr+=1   
            scores.append(score)
        #print(str(class_score//len(test_loader)))
        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)

        #print(gt_list)

        img_roc_auc,y_score=computeAUROC(scores,gt_list,self.obj," "+self.distillType)
        fpr,tpr,ths=computeROCcurve(y_true,y_score)
        #print(type(fpr))
        #writer.add_scalars('ROC',tpr,fpr)
        writer.add_pr_curve("ROC "+str(self.obj),np.array(y_true),np.array(y_score),None,len(ths))
        metric=BinaryConfusionMatrix()
        metric.update(torch.tensor(y_score),torch.tensor(y_true))
        print(metric.compute())
        #( (true positive, false negative)
        # (false positive, true negative) )
    
        return img_roc_auc
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("/home/christianjaspert/masterthesis/DistillationAD/config.yaml")
    

    if data['phase'] == "train":
        trainer = NetTrainer(data,device,False,None)
        trainer.train()
        trainer.test()
    elif data['phase'] == "test":
        trainer = NetTrainer(data,device,False,None)
        trainer.test()
    elif data['phase'] == "statistics_epochs":
        
        lrlist=[]
        epmax=100
        for epochs in [1,5,10,20,30,50,100]:
            trainer = NetTrainer(data,device,epochs)
            mylist=trainer.train()
            for i in range(epmax+1-len(mylist)):
                if len(mylist)<epmax+1:
                    mylist.append([0,0])
            #print(mylist)
            lrlist.append(mylist)

        for _ in range(len(lrlist[len(lrlist)-1])):
            writer.add_scalars('losses for different num_epochs '+str(data['obj']),{'tl e1':lrlist[0][_][0],
                                                                      'vl e1':lrlist[0][_][1],
                                                                      'tl e5':lrlist[1][_][0],
                                                                      'vl e5':lrlist[1][_][1],
                                                                      'tl e10':lrlist[2][_][0],
                                                                      'vl e10':lrlist[2][_][1],
                                                                      'tl e20':lrlist[3][_][0],
                                                                      'vl e20':lrlist[3][_][1],
                                                                      'tl e30':lrlist[4][_][0],
                                                                      'vl e30':lrlist[4][_][1],
                                                                      'tl e50':lrlist[5][_][0],
                                                                      'vl e50':lrlist[5][_][1],
                                                                      'tl e100':lrlist[6][_][0],
                                                                      'vl e100':lrlist[6][_][1],
                                                                      },_+1)
    elif data['phase']=="statistic_datasets":
        lrlist=[]
        epmax=5
        for dataset in [[]]:
            trainer = NetTrainer(data,device,epochs)
            mylist=trainer.train()
            for i in range(epmax+1-len(mylist)):
                if len(mylist)<epmax+1:
                    mylist.append([0,0])
            #print(mylist)
            lrlist.append(mylist)
        
    
    else:
        print("Phase argument must be train or test.")

