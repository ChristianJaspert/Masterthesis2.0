import torch.nn as nn
from torch.nn import ConvTranspose2d 
from models.DBFAD.utilsModel import  conv3BnRelu, conv1BnRelu,BasicBlockDe, Attention,AttentionMinus1, Attention2,Attention1,Attention3, Attention4, Attention5,Attention6,Attention7
from models.sspcab import SSPCAB
import sys
from utils.util import readYamlConfig
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("runs")

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class ReverseStudent(nn.Module):
    def __init__(self, block, layers,groups=1,DG=False):
        super(ReverseStudent, self).__init__()
        data=readYamlConfig("/home/christianjaspert/masterthesis/DistillationAD/config.yaml")
        
        self.DG=DG
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 512  
        
        self.dilation = 1
            
        self.groups = groups
        self.base_width = 64


        #Blocks for the feature fusion{
        self.block1 = nn.Sequential(
            
            conv1BnRelu(64, 128, stride=2, padding=0), 
            conv1BnRelu(128, 256, stride=2, padding=0)
        )
        self.block2 = nn.Sequential(
            conv1BnRelu(64, 128, stride=2, padding=0),
            conv1BnRelu(128, 256, stride=2, padding=0),
        )
        self.block3 = nn.Sequential(
            conv1BnRelu(128, 256, stride=2, padding=0)
        )
        self.block4 = nn.Sequential(
            conv1BnRelu(256, 256, stride=1, padding=0)
        )
        #}
        #BottleNeck{
        self.block5 = nn.Sequential(
            conv1BnRelu(256, 256, stride=1, padding=0),
            conv1BnRelu(256, 256, stride=1, padding=0),
            conv1BnRelu(256, 512, stride=2, padding=0),
            SSPCAB(512)
        )
        attentionswitch=data['myworkswitch']
        if attentionswitch:
            l=data['attentionlayer']
            if l==1:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention1(512, 512)
            )
            if l==2:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention2(512, 512)
            )
            if l==3:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention3(512, 512)
            )
            if l==4:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention4(512, 512)
            )
            if l==5:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention5(512, 512)
            )
            if l==6:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention6(512, 512)
            )
            if l==7:
                self.AttBlock = nn.Sequential(
                    #Attention2(512, 512)
                    Attention7(512, 512)
            )
        else:
            self.AttBlock = nn.Sequential(
                Attention2(512, 512)
            )

        #}

        #Distillation residual layer between Teacher and student
        distillswitch=False #data['myworkswitch']
        if distillswitch:
            #l=-1,1,2,3,4,5,6,7,31
            l=data['distillationlayer']
            if l==-1:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        AttentionMinus1(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        AttentionMinus1(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        AttentionMinus1(256, 256)
                    )
            elif l==1:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention1(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention1(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention1(256, 256)
                    )
            elif l==2:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention2(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention2(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention2(256, 256)
                    )
            elif l==3:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention3(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention3(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention3(256, 256)
                    )
            elif l==4:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention4(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention4(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention4(256, 256)
                    )
            elif l==5:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention5(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention5(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention5(256, 256)
                    )
            elif l==6:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention6(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention6(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention6(256, 256)
                    )
            elif l==7:
                if not self.DG:
                    self.residualLayer0_1 = nn.Sequential(
                        conv3BnRelu(64, 64, stride=1, padding=1),
                        Attention7(64, 64)
                    )
                    self.residualLayer1_2 = nn.Sequential(
                        conv3BnRelu(64, 128, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention7(128, 128)
                    )
                    self.residualLayer2_3 = nn.Sequential(
                        conv3BnRelu(128, 256, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Attention7(256, 256)
                    )
            elif l==31:
                self.residualLayer0_1 = nn.Sequential(
                    conv3BnRelu(64, 64, stride=1, padding=1),
                    Attention(64, 64)
                )
                self.residualLayer1_2 = nn.Sequential(
                    conv3BnRelu(64, 128, stride=1, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    Attention(128, 128)
                )
                self.residualLayer2_3 = nn.Sequential(
                    conv3BnRelu(128, 256, stride=1, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    Attention(256, 256)
                )
        else:
            if not self.DG:
                self.residualLayer0_1 = nn.Sequential(
                    conv3BnRelu(64, 64, stride=1, padding=1),
                    Attention(64, 64)
                )
                self.residualLayer1_2 = nn.Sequential(
                    conv3BnRelu(64, 128, stride=1, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    Attention(128, 128)
                )
                self.residualLayer2_3 = nn.Sequential(
                    conv3BnRelu(128, 256, stride=1, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    Attention(256, 256)
                )

        #print("lb",layers,block)
        #Normal layers of Student
        self.layer0 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: BasicBlockDe, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        #print(planes*block.expansion)
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    #here first the feature fusion and then the bottleneck are 
    #calculated and give as output the input for the studetnt decoder
    #x is the tensor of features_t from the teacher
    def _forward_impl(self, x):
        #print("forward_impl_revResid",x[0].shape,x[0][0][0][0])

        out1 = self.block1(x[0])
        out2 = self.block2(x[1])
        out3 = self.block3(x[2])
        out4 = self.block4(x[3])
        out = (out1 + out2 + out3 + out4)
        out = self.block5(out)
        out = self.AttBlock(out)
        
        
        if not self.DG:
            resi1 = self.residualLayer0_1(x[0])
            resi2 = self.residualLayer1_2(x[1])
            resi3 = self.residualLayer2_3(x[2])

        #!!!!!!!!!!!!!!!!!!!!!!these things were after the first feature_x calculation:

        #print(x[2],"x[2]")
        #print(self.residualLayer2_3(x[2]).shape,"residualLayer")
        #print(resi3.shape,"resi3")
        #print(out.shape,"out")
        
        #print(feature_x.shape,"feature_X=layer0(out)")
        #print(self.layer0)
        feature_x = self.layer0(out)
        feature_x = feature_x + resi3 if not self.DG else feature_x
        feature_a = self.layer1(feature_x)  
        feature_a = feature_a+resi2 if not self.DG else feature_a
        feature_b = self.layer2(feature_a)  
        feature_b = feature_b+resi1 if not self.DG else feature_b
        feature_c = self.layer3(feature_b)

        return feature_c, feature_b, feature_a, feature_x

    def forward(self, x):
        x, x1, x2, x3 = self._forward_impl(x)
        return x, x1, x2, x3


def reverse_student(block, layers,DG, **kwargs):
    model = ReverseStudent(block, layers,DG=DG, **kwargs)
    return model


def reverse_student18(DG=False, **kwargs):
    return reverse_student(BasicBlockDe, [2, 2, 2, 2],DG=DG,
                           **kwargs)