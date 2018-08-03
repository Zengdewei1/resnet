import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def _weights_init(m):
    if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
        init.kaiming_normal_(m.weight)
        
class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(BasicBlock,self).__init__()
        self.mainLayer=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False),
        nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(out_channel))
        self.shortcut=nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channel))
    
    def forward(self,input):
        out=self.mainLayer(input)
        out+=self.shortcut(input)
        out=F.relu(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self,BasicBlock):
        super(ResNet,self).__init__()
        self.in_channel=64
        self.conv1=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(64),nn.ReLU())
        self.layer1=self.make_layer(BasicBlock,64,3,stride=1)
        self.layer2=self.make_layer(BasicBlock,128,4,stride=2)
        self.layer3=self.make_layer(BasicBlock,256,6,stride=2)
        self.layer4=self.make_layer(BasicBlock,512,3,stride=2)
        self.Dropout=nn.Dropout(0.5)
        self.fc=nn.Linear(512,10)
        
        self.apply(_weights_init)
    def make_layer(self,block,channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks - 1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_channel,channels,stride))
            self.in_channel=channels
        return nn.Sequential(*layers)
    def forward(self,input):
        output=self.conv1(input)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=F.avg_pool2d(output,4)
        output=output.view(output.size(0),-1)
        output=self.fc(self.Dropout(output))
        return output

def ResNet34():
        return ResNet(BasicBlock)