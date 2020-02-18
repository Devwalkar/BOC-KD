import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ['BIO_ResNet', 'BIO_ResNet20', 'BIO_ResNet32', 'BIO_ResNet44', 'BIO_ResNet56', 'BIO_ResNet110', 'BIO_ResNet1202']

def depth_channel_computer(no_blocks = 3,
                           no_students = 4,
                           original_channels = [128,256,512]
                           ):

    Depth_channels_list = []

    if no_blocks==1:
        for j in range(no_students):
            Depth_channels_list.append([original_channels[0],
                                        original_channels[1],
                                        int(original_channels[2]*(float(no_students-j)/no_students))
                                        ])
    
    elif no_blocks==2:
        for j in range(no_students):
            Depth_channels_list.append([original_channels[0],
                                        int(original_channels[1]*(float(no_students-j)/no_students)),
                                        int(original_channels[2]*(float(no_students-j)/no_students))
                                        ])       

    elif no_blocks==3:
        for j in range(no_students):
            Depth_channels_list.append([int(original_channels[0]*(float(no_students-j)/no_students)),
                                        int(original_channels[1]*(float(no_students-j)/no_students)),
                                        int(original_channels[2]*(float(no_students-j)/no_students))
                                        ])    

    else:
        raise IndexError("No blocks can only be 1,2 or 3")

    return Depth_channels_list


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        return out

class ResNet_Student(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                depth_channels=[16,32,64]
                ):
        super(ResNet_Student, self).__init__()
        self.in_planes = 16

        self.layer1 = self._make_layer(block, depth_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, depth_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, depth_channels[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(depth_channels[2], num_classes)
        self.dropout = nn.Dropout(p=0.3)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out = F.avg_pool2d(x3, x3.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,[x1,x2,x3]

class BIO_ResNet(nn.Module):

    # Overall resnet student framework buiding function which integrates base resnet with 
    # Student Resnet models

    def __init__(self, block, layers, num_classes=1000,
                 Base_freeze= False,
                 no_students = 4,
                 no_blocks = 3,             # Select only from 3,2,1
                 parallel = False,
                 gpus = [0,1],
                 Common_Base = False,
                 Single_model = 0                         
                 ):

        super(BIO_ResNet, self).__init__()

        Depth_channels_list = depth_channel_computer(no_blocks=no_blocks,
                                                         no_students=no_students,
                                                         original_channels=[16,32,64])

        self.no_students = no_students
        self.no_blocks = no_blocks
        self.pretrain_mode = False
        self.single_model_mode = False
        self.Common_Base = Common_Base
        self.single_model = Single_model

        if Common_Base:
            print("---------- Passing collective gradients through Common Base")

        # Initializing the common base ResNet model

        self.BaseNet = BaseNet()

        self.BaseNet = self.BaseNet.to(device) if not parallel else torch.nn.DataParallel(self.BaseNet.to(device),
                                                                                               device_ids =gpus)

        if Base_freeze:
            self.base_freezer()

        # Initializing student models

        self.student_models = []

        for depth_channels in Depth_channels_list:

            Student_M = ResNet_Student(block=block,
                                       num_blocks=layers, 
                                       num_classes=num_classes, 
                                       depth_channels=depth_channels
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
                Final_out,Inter_reps = self.student_models[i](x_copy) if self.Common_Base else self.student_models[i](x)
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


def BIO_ResNet20(**kwargs):
    model = BIO_ResNet(BasicBlock, [3, 3, 3],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model


def BIO_ResNet32(**kwargs):
    model = BIO_ResNet(BasicBlock, [5, 5, 5],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model

def BIO_ResNet44(**kwargs):
    model = BIO_ResNet(BasicBlock, [7, 7, 7],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model

def BIO_ResNet56(**kwargs):
    model = BIO_ResNet(BasicBlock, [9, 9, 9],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model

def BIO_ResNet110(**kwargs):
    model = BIO_ResNet(BasicBlock, [18, 18, 18],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model

def BIO_ResNet1202(**kwargs):
    return BIO_ResNet(BasicBlock, [200, 200, 200],**kwargs)
    parameter_model_counter(model.BaseNet,model.student_models)
    return model
