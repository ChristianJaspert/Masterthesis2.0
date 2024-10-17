import os
import torch
from torch.utils.data import Dataset
#from torchvision import transforms as T
from torchvision.transforms import v2 as T
import cv2
import numpy as np
import glob
import sys

class MVTecDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None,crop_size=None,phase="train",croppingfactor=4,cropping=True,augmentation=False):
        self.root_dir = root_dir
        self.croppingfactor=croppingfactor
        self.cropping=cropping
        image_extensions = ['png', 'tif', 'tiff', 'jpg', 'jpeg']
        pattern = f"{root_dir}/*/*" + '/'.join(f"*.{ext}" for ext in image_extensions)

        self.images = sorted(glob.glob(root_dir+"/*/*.png"))

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.augmentation=False #augmentation

        self.resize_shape=resize_shape
        if (crop_size==None):
            crop_size=min(resize_shape[0],resize_shape[1])
        #self.transform=T.Compose([T.CenterCrop(crop_size),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.transform=T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.trainingtransform=T.Compose([
            T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
            #T.RandomAdjustSharpness(sharpness_factor=.5),
            T.RandomAffine(degrees=180),
            T.ColorJitter(brightness=.5,contrast=.4)
            #T.RandomResize(min_size=10,max_size=100),
            #T.RandomResizedCrop(self.resize_shape)
            ])
        self.phase=phase
        
    def __len__(self):
        if self.phase=="test":
            return len(self.images)
        else:
            return len(self.image_paths)

    def transform_image(self, image_path):
        
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #shape(h,w,rgb)
        if self.cropping:
            #print(image,self.croppingfactor)
            cropped_img=self.crop_img(image,self.croppingfactor)
            
            concat_img=np.zeros((3,self.resize_shape[1]*self.croppingfactor,self.resize_shape[0]*self.croppingfactor))
            i=0
            for image in cropped_img:
                if self.resize_shape != None:
            # cv2.resize(dsize(width,height))
            #resize_shape=(height,width)
                    image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
                heigth,width=image.shape[0], image.shape[1]        
                image = np.array(image).reshape((heigth,width, 3)).astype(np.float32)/ 255.0
                
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))
                image=np.asarray(self.transform(torch.from_numpy(image)))
                w=int(i/self.croppingfactor)
                h=int(i%self.croppingfactor)
                concat_img[:,h*heigth:(h+1)*heigth,w*width:(w+1)*width]=image
                #print("h",int(i%self.croppingfactor),"w",int(i/self.croppingfactor))
                i+=1
            image=concat_img
        else:
            if self.resize_shape != None:
                # cv2.resize(dsize(width,height))
                #resize_shape=(height,width)
                image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
                
            image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)/ 255.0
            
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image=self.transform(torch.from_numpy(image))
            if self.augmentation:
                #print("before",image)
                image=self.trainingtransform(image)
                #print("after",image)
                #sys.exit()
            image=np.asarray(image)
            #print("mvtec image",image.dtype)
        if self.phase=="test":
            return image   
        else:
            return image


 
    def __getitem__(self, idx):
        if self.phase=="test":
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_path = self.images[idx]
            dir_path, file_name = os.path.split(img_path)
            base_dir = os.path.basename(dir_path)
            image = self.transform_image(img_path)
            #print("image dtype",image.dtype)
            if base_dir == 'good':
                has_anomaly = np.array([0], dtype=np.int64)
            else:
                has_anomaly = np.array([1], dtype=np.int64)
            sample = {'imageBase': image, 'has_anomaly': has_anomaly, 'idx': idx, 'file_name':file_name}
        else:
            idx = torch.randint(0, len(self.image_paths), (1,)).item()
            image = self.transform_image(self.image_paths[idx])
            sample = {'imageBase': image}
        return sample
    
    def crop_img(self,img,croppingfactor): #shape(h,w,rgb)=(0,1,2)
        '''
        takes torch tensor with one sample image (shape: [1,rgb,height,width])
        and a croppingfactor (eg. 4) that says what fraction of the image measurements the cropped ones have
        (eg 1/4 of length and width -> 16 cropped images)
        returns a list of cropped images in the following order:
        first column of image then second and so on
        '''
        height,width = img.shape[0],img.shape[1] 
        w_rest=width%croppingfactor
        h_rest=height%croppingfactor
        torch_img_rest = img[h_rest:height,w_rest:width,:]
        height,width=torch_img_rest.shape[0],torch_img_rest.shape[1] 
        list_cropped_torch_images=[]
        for w in range(croppingfactor):
            for h in range(croppingfactor):
                torch_img_cropped=torch_img_rest[int(h*(height/croppingfactor)):int((h+1)*height/croppingfactor),int(w*width/croppingfactor):int((w+1)*width/croppingfactor),:]
                #img_cropped=img_rest.crop((w*(width/factor),h*height/factor,(w+1)*width/factor,(h+1)*height/factor))
                list_cropped_torch_images.append(torch_img_cropped)
        return list_cropped_torch_images
