import timm #collection of pretrained image NNs
import torch
import torch.nn as nn
import torch.nn.functional as F

class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[2]
    ):
        super(teacherTimm, self).__init__()  
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        self.feature_extractor.eval() 
        #freezeing of teacher parameters:
        for param in self.feature_extractor.parameters():
            #param=param.type(torch.float64)
            #print("models/teacher.py",param.dtype)
            param.requires_grad = False   
        
    def forward(self, x):
        features_t = self.feature_extractor(x)
    
        return features_t
    