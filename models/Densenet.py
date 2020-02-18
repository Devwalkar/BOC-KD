import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models
from torch.autograd import Variable 
import numpy as np
import sys
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Growth_rate_computer(no_students = 4,
                         original_growth_rate =32
                           ):

    Growth_rate_list = []

    for j in range(no_students):
            Growth_rate_list.append(int(original_growth_rate*(float(no_students-j)/no_students))
                                   )    

    return Growth_rate_list


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class BaseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(BaseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)

        self.in_Channels = nChannels

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        return out


class DenseNet_Student(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck,nChannels):
        super(DenseNet_Student, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.trans1(self.dense1(x))
        x_2 = self.trans2(self.dense2(x_1))
        x_3 = self.dense3(x_2)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(x_3)), 8))
        out = self.fc(out)
        return out,[x_1,x_2,x_3]


class BIO_DenseNet(nn.Module):

    def __init__(self,growthRate, depth, reduction, bottleneck, num_classes = 1000, 
                no_blocks = 3, no_students = 4, parallel=False,gpus=[0,1], 
                Base_freeze = False, Common_Base=False,
                Single_model=0):

        super(BIO_DenseNet, self).__init__()

        # Initializing the common base resent model

        self.BaseNet = BaseNet(growthRate=growthRate,
                               depth=depth,
                               reduction=reduction,
                               nClasses=num_classes,
                               bottleneck=bottleneck)

        self.BaseNet = self.BaseNet.to(device) if not parallel else torch.nn.DataParallel(self.BaseNet.to(device),
                                                                                          device_ids =gpus)
        if Base_freeze:
            self.base_freezer()
                                                                                         
        growth_rate_list = Growth_rate_computer(no_students=no_students,
                                                original_growth_rate=growthRate
                                                )

        self.pretrain_mode = False
        self.single_model_mode = False
        self.no_students = no_students
        self.no_blocks = no_blocks
        self.Common_Base = Common_Base
        self.single_model = Single_model
        self.student_num_features = self.BaseNet.module.in_Channels if parallel else  self.BaseNet.in_Channels

        if Common_Base:
            print("---------- Passing collective gradients through Common Base")
        # Initializing student models

        self.student_models = []

        for growth_rate in growth_rate_list:

            Student_M = DenseNet_Student(growthRate=growth_rate,
                                         depth=depth,
                                         reduction=reduction,
                                         nClasses=num_classes,
                                         bottleneck=bottleneck,
                                         nChannels = self.student_num_features
                                         )

            if not parallel:
                self.student_models.append(Student_M.to(device))
            else:
                self.student_models.append(torch.nn.DataParallel(Student_M.to(device),device_ids=gpus))
        
        # Initializing contribution weights for teacher output

        self.weights = Variable(torch.Tensor((float(1)/self.no_students)*np.ones([1,self.no_students,1])),
                                requires_grad= True).to(device)

    def base_freezer(self):

        print("----------- Freezing common base model")

        for m in self.BaseNet.parameters():
            if m.requires_grad:
                m.requires_grad = False

    def pretrain(self):
        self.pretrain_mode = True

    def Single_mode(self):
        self.single_model_mode = True

    def student_version(self):
        self.pretrain_mode = False

    def _forward_teacher_pretrain(self, x):

        x = self.BaseNet(x)

        Teacher_out,_ = self.student_models[0](x)         

        return Teacher_out

    def _forward_student_model(self, x):

        x = self.BaseNet(x)

        Teacher_out,_ = self.student_models[int(self.single_model)](x)         

        return Teacher_out

    def _forward_student(self, x):

        x = self.BaseNet(x)

        x_copy = Variable(x.clone(),requires_grad=False)

        Student_final_outs = []
        Student_intermmediate_reps = [[] for _ in range(self.no_blocks)]

        for i in range(self.no_students):
            if i==0:
                Final_out,Inter_reps = self.student_models[0](x)
                Combined_student_outs = Final_out.unsqueeze(1)
                Inter_reps = [single_rep.detach() for single_rep in Inter_reps]
            else:
                Final_out,Inter_reps = self.student_models[i](x) if self.Common_Base else self.student_models[i](x_copy)
                Combined_student_outs = torch.cat((Combined_student_outs,Final_out.unsqueeze(1)),dim=1) # Combined_student_outs shape : (Batch_size,no_students,num_classes) 

            Student_final_outs.append(Final_out)
            
            for j in range(self.no_blocks):
                Student_intermmediate_reps[j].append(Inter_reps[j])

        Teacher_out = torch.sum((self.weights*Combined_student_outs),dim=1).squeeze(1)               # Teacher out shape : (Batch_size, num_classes)

        Student_final_outs = [Teacher_out] + Student_final_outs           

        return Student_final_outs,Student_intermmediate_reps

    def forward(self, x):
        if self.pretrain_mode:
            return self._forward_teacher_pretrain(x)
        elif self.single_model_mode:
            return self._forward_student_model(x)
        else:
            return self._forward_student(x)

def count_parameters(model):

    total = 0
    trainable = 0
    for param in model.parameters():
        temp = param.numel()
        total += temp
        if param.requires_grad:
            trainable += temp

    return (total,trainable)

def parameter_model_counter(base_model,student_models):

    # Function to count individual student model parameters

    Base_model_count = count_parameters(base_model)
    Student_model_count = [count_parameters(model) for model in student_models]

    Original_model_params = Base_model_count[0]+Student_model_count[0][0]

    Total_params,Total_trainable_params = Base_model_count[0],Base_model_count[1]
    print("\nParameter count -->")
    for i,stu_count in enumerate(Student_model_count):
        print("Student {}: Total {} | Trainable: {} Compress_ratio: {:.4f}".format(i+1,Base_model_count[0]+stu_count[0],
                                                                               Base_model_count[1]+stu_count[1],
                                                            float(Base_model_count[0]+stu_count[0])/Original_model_params))
        Total_params+=stu_count[0]
        Total_trainable_params+=stu_count[1]
    
    print("Overall Teacher: Total: {} | Trainable: {}\n".format(Total_params,Total_trainable_params))

def BIO_Densenet_12_100(**kwargs):

    model = BIO_DenseNet(growthRate=12,depth=100,reduction=0.5,bottleneck=True,**kwargs)

    parameter_model_counter(model.BaseNet,model.student_models)

    return model


def BIO_Densenet_24_250(**kwargs):

    model = BIO_DenseNet(growthRate=24,depth=250,reduction=0.5,bottleneck=True,**kwargs)

    parameter_model_counter(model.BaseNet,model.student_models)

    return model


def BIO_Densenet_40_190(**kwargs):

    model = BIO_DenseNet(growthRate=40,depth=190,reduction=0.5,bottleneck=True,**kwargs)

    parameter_model_counter(model.BaseNet,model.student_models)

    return model