import os
import sys
from pympler import asizeof
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time
from datetime import timedelta,datetime
import numpy as np
import torch
import pickle
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from utils.util import  AverageMeter,readYamlConfig,set_seed,convert_secs2time
from utils.functions import (
    cal_loss,
    cal_anomaly_maps,
    #th_img,
    write_in_csv,
    #get_classification,
    crop_torch_img,
    #img_transposetorch2nparr,
    concat_hm,
    save_csv_hm,
    generate_result_path,
    save_log_csv,
    augmented_scores
    
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
        #inittime=time.time()
        loadModels(self)
        #print("Load models time(h:m:s): ",convert_secs2time(inittime-time.time()))
        loadDataset(self)
        #print("Load models and dataset time(h:m:s): ",convert_secs2time(inittime-time.time()))
        #sys.exit()
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
        

    def train(self,):
        print("training " + self.obj)
        self.student.train() 
        best_score = None
        start_time = time.time()
        ttime=start_time
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        losslist=[]
        for _ in range(1, self.num_epochs + 1):
            
            losses = AverageMeter()
            loss_epoch=0
            i=0
            for sample in self.train_loader:
                

                i=i+1
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
                #print(asizeof.asizeof(self.optimizer.step()))
                #print(trainer.device)
                #print(torch.cuda.memory_summary(trainer.device))
                epoch_bar.set_postfix({"Loss": loss.item(),"e":_,"i":i,"tni":len(self.train_loader),"EpTav":convert_secs2time(epoch_time.avg)}) #,"Time/epoch":convert_secs2time(etime)})
                epoch_bar.update()
                #0print(loss_epoch.item())
            if self.write:
                writer.add_scalar('training loss '+str(self.obj)+"-"+self.myworklabel,loss_epoch,_)
               
            
            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
            if self.write:
                writer.add_scalars('losses lr: '+str(self.obj)+"-"+self.myworklabel+": "+str(self.lr),{'training loss':loss_epoch.item(),'validation loss':val_loss},_)
            losslist.append([loss_epoch.item(),val_loss])
            epoch_time.update(time.time() - start_time)
            #print("epoch time",convert_secs2time(epoch_time.val))
            epoch_bar.set_postfix({"Loss": loss.item(),"EpT":convert_secs2time(epoch_time.val)}) #,"Time/epoch":convert_secs2time(etime)})
            epoch_bar.update
            start_time = time.time()
        traintime=time.time()-ttime
        self.traintime=traintime
        print("training time in sec", convert_secs2time(traintime))
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
        if self.obj=="carpet":
            test_path=self.data_path+"/"+self.obj+"/test/"
            self.cropping=False
        else:
            if self.cropping:
                cropfolder="not_cropped"
            else:
                cropfolder="cropped"
            if self.handmade:
                testfolder="test_" + self.handmadetype
            else:
                testfolder="test"
            if self.plotting_hm:
                testfolder=testfolder+"_plots"
            test_path=self.data_path+"/"+self.obj + "/"+testfolder+"/"+cropfolder+"/"            
            # if self.cropping:
            #     if self.handmade:
            #         if self.handmadetype=="SR":
            #             test_path=self.data_path+"/"+self.obj+"/test_SR/not_cropped/"
            #         elif self.handmadetype=="FW":
            #             test_path=self.data_path+"/"+self.obj+"/test_FW/not_cropped/"
            #     else:
            #         test_path=self.data_path+"/"+self.obj+"/test/not_cropped/"
            # else:
            #     if self.handmade:
            #         if self.handmadetype=="SR":
            #             test_path=self.data_path+"/"+self.obj+"/test_SR/cropped/"
            #         elif self.handmadetype=="FW":
            #             test_path=self.data_path+"/"+self.obj+"/test_FW/cropped/"
            #     else:
            #         test_path=self.data_path+"/"+self.obj+"/test/cropped/"
        
        test_dataset = MVTecDataset(
            root_dir=test_path,
            resize_shape=[self.img_resize_h,self.img_resize_w],
            crop_size=[self.img_cropsize,self.img_cropsize],
            phase='test',
            croppingfactor=self.croppingfactor,
            cropping=self.cropping
        )

        tag="idx: "+str(test_dataset.__getitem__(0).get("idx"))+"  has anomaly: "+str(test_dataset.__getitem__(0).get("has_anomaly"))+"  filename: "+str(test_dataset.__getitem__(0).get("file_name"))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        progressBar = tqdm(test_loader)
        
        
        scores = [] #np.asarray([])
        gt_list = [] #ground truth labels
        #y_true=[]
        hm_dir_basis,csv_path,test_timestamp=generate_result_path(self)
        pred_scores=[]
        minpositive=1
        maxnegative=0

        for sample in test_loader:
            
            label=sample['has_anomaly']
            #y_true.append(label.cpu().numpy()[0][0])
            image = sample['imageBase'].to(self.device)
            gt_list.extend(label.cpu().numpy())
            
            pickleuse=False
            if not pickleuse:
                with torch.set_grad_enabled(False):
                    if trainer.cropping:
                        #print("cropping")
                        th=0.4 #in % between 0 and 1
                        area_th=20000
                        cropped_scores=[]
                        i=1
                        for cropped_img in crop_torch_img(image,self.croppingfactor,self.overlapfactor):
                            #features_s, features_t = infer(self,cropped_img)
                            aug_scores=augmented_scores(self,cropped_img,str(sample["file_name"]))
                            #score=cal_anomaly_maps(features_s,features_t,self.img_cropsize,self.norm)
                            score=np.mean(aug_scores, axis=0)
                            cropped_scores.append(score)
                            progressBar.set_postfix({"cropped img": i}) #,"Time/epoch":convert_secs2time(etime)})
                            #progressBar.update()
                            i+=1
                    

                        score=concat_hm(image,cropped_scores,self.croppingfactor,self.overlapfactor)
                    else:
                        th=0.875 #in % between 0 and 1
                        area_th=100
                        #print("not cropping")
                        #features_s, features_t = infer(self,image)  
                        #score =cal_anomaly_maps(features_s,features_t,self.img_cropsize,self.norm)
                        aug_scores=augmented_scores(self,image,str(sample["file_name"]))
                        score=np.mean(aug_scores, axis=0)
                

                pmaxth=0.0003
                if False:
                    predscore,hmnumanomalypixel=save_csv_hm(sample,score,hm_dir_basis,self.hm_sorting,csv_path,th,area_th,self.blendfactor,pmaxth)
                #pred_scores.append(predscore)
                    predictioncsvarr=[label.cpu().numpy()[0][0],predscore,hmnumanomalypixel,self.param_str,self.obj,str(datetime.now().hour)+"_"+str(datetime.now().minute),self.save_path]
                    write_in_csv('/home/christianjaspert/masterthesis/DistillationAD/predictions.csv',predictioncsvarr)

                    if label.cpu().numpy()==1:
                        if minpositive>predscore:
                            minpositive=predscore
                    else:
                        if maxnegative<predscore:
                            maxnegative=predscore
                #print(score.shape)
                #scores=np.append(scores,score)

                scores.append(score)
                # print(score)
                # if np.any(np.isnan(score)==True):
                #     print(score)
                #     print(sample["file_name"])

            progressBar.update() 
            
            
        pmaxth=0.0003#####################
        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        #pred_scores=np.array(pred_scores)
        groundtruth=np.array(gt_list[:,0])
        

        
        if False:
            writeread="write"
            if writeread=="write":
                with open('scores.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
            else: 
                with open('scores.pickle', 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                    scores=pickle.load(f)
                    print(scores)

        print("Done writing pickle")
        img_roc_auc,y_score,optmatrix,th=computeAUROC(self,scores,gt_list,self.obj+("-"+self.myworklabel if self.myworkswitch else "")," "+self.distillType)
        save_log_csv(self,optmatrix,th,test_timestamp,img_roc_auc)
        
        if trainer.cropping:
            area_th=20000
        else:
            area_th=100
        #sys.exit()
        s=0  
        for sample in test_loader:
            score=scores[s]
            binth=.7
            save_csv_hm(sample,score,hm_dir_basis,self.hm_sorting,csv_path,binth,area_th,self.blendfactor,pmaxth)
            #predictioncsvarr=[label.cpu().numpy()[0][0],predscore,self.param_str,self.obj,str(datetime.now().hour)+"_"+str(datetime.now().minute),self.save_path]
            #write_in_csv('/home/christianjaspert/masterthesis/DistillationAD/predictions.csv',predictioncsvarr)
            # if label.cpu().numpy()==1:
            #     if minpositive>predscore:
            #         minpositive=predscore
            # else:
            #     if maxnegative<predscore:
            #         maxnegative=predscore
            s+=1
        
        #confusion matrix:
        #              predicted
        #actual     ((true positive, false negative)
        #            (false positive, true negative) )
    
        return img_roc_auc
    
    def score(self,inputimage):
        features_s, features_t = infer(self,inputimage)
        score=cal_anomaly_maps(features_s,features_t,self.img_cropsize,self.norm)
        return score
    


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
    # elif data['phase'] == "statistics_epochs":
        
    #     lrlist=[]
    #     epmax=100
    #     for epochs in [1,5,10,20,30,50,100]:
    #         trainer = NetTrainer(data,device,epochs)
    #         mylist=trainer.train()
    #         for i in range(epmax+1-len(mylist)):
    #             if len(mylist)<epmax+1:
    #                 mylist.append([0,0])
    #         #print(mylist)
    #         lrlist.append(mylist)

    #     for _ in range(len(lrlist[len(lrlist)-1])):
    #         writer.add_scalars('losses for different num_epochs '+str(data['obj']),{'tl e1':lrlist[0][_][0],
    #                                                                   'vl e1':lrlist[0][_][1],
    #                                                                   'tl e5':lrlist[1][_][0],
    #                                                                   'vl e5':lrlist[1][_][1],
    #                                                                   'tl e10':lrlist[2][_][0],
    #                                                                   'vl e10':lrlist[2][_][1],
    #                                                                   'tl e20':lrlist[3][_][0],
    #                                                                   'vl e20':lrlist[3][_][1],
    #                                                                   'tl e30':lrlist[4][_][0],
    #                                                                   'vl e30':lrlist[4][_][1],
    #                                                                   'tl e50':lrlist[5][_][0],
    #                                                                   'vl e50':lrlist[5][_][1],
    #                                                                   'tl e100':lrlist[6][_][0],
    #                                                                   'vl e100':lrlist[6][_][1],
    #                                                                   },_+1)
    # elif data['phase']=="statistic_datasets":
        # lrlist=[]
        # epmax=5
        # for dataset in [[]]:
        #     trainer = NetTrainer(data,device,epochs)
        #     mylist=trainer.train()
        #     for i in range(epmax+1-len(mylist)):
        #         if len(mylist)<epmax+1:
        #             mylist.append([0,0])
        #     #print(mylist)
        #     lrlist.append(mylist)
        
    
    else:
        print("Phase argument must be train or test.")

