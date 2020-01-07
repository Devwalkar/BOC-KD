from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KL_Loss(nn.Module):
    def __init__(self,T=3):

        # KL loss module for comparing teacher and student prediction distributions

        super(KL_Loss, self).__init__()
        self.T = T

    def forward(self, teacher_pred, student_pred):

        # teacher_pred shape: (Batch_size,num_classes)
        # student_pred shape: (Batch_size,num_classes)

        student_pred = F.log_softmax(student_pred/self.T,dim=1)
        teacher_pred = F.softmax(teacher_pred/self.T,dim=1)

        teacher_pred = teacher_pred + 10**(-7)
        teacher_pred = Variable(teacher_pred.data.to(device),requires_grad=False)
        KL_loss = (self.T**2) * ((teacher_pred*(teacher_pred.log()-student_pred)).sum(1).sum()/teacher_pred.size()[0])

        return KL_loss


class Intermmediate_loss(nn.Module):

    def __init__(self, no_students = 4,
                       no_blocks = 3, 
                       Loss_module = 'MSE'
                ):

        # Intermmediate loss module for blockwise representation comparision of pseudo teacher with students
         
        super(Intermmediate_loss,self).__init__()
        self.no_students = no_students
        self.no_blocks = no_blocks
        self.loss_module = getattr(nn,Loss_module)()
    
    def forward(self,intermmediate_maps):

        # intermmediate_maps shape : [list of maps per block level 1 ,list of maps per block level 2, .... ]
        # Each list of maps shape  : [Pseudo-teacher map, Student 1 map, ....]
        # Each map shape           : (Batch_size,no_channels,width,height)

        Total_loss = 0

        for i in range(self.no_blocks):
            Pseudo_Teacher_pred = intermmediate_maps[i][0]

            Pseudo_Teacher_pred = Pseudo_Teacher_pred.mean(1).squeeze(1)
            
            for j in range(1,self.no_students):
                intermmediate_map = intermmediate_maps[i][j].mean(1).squeeze(1)
                Total_loss += self.loss_module(Pseudo_Teacher_pred,intermmediate_map)

                
        return Total_loss


class Normal_Loss(nn.Module):

    def __init__(self,
                 loss_module = "CrossEntropyLoss"
                 ):

        # Loss module for collective normal loss computation for the students and the teacher models

        super(Normal_Loss,self).__init__()

        self.loss_module = getattr(nn,loss_module)()

    def forward(self,preds,labels):

        # preds shape            :  [preds of teacher model, preds of Student model 1]
        # individual preds shape :  (batch_size, no_classes)
        # Labels shape           :  (Batch_size, no_classes)

        Total_loss = []

        for model_pred in preds:

                Total_loss.append(self.loss_module(model_pred,labels))

        return Total_loss


class Combined_Loss(nn.Module):

    def __init__(self,
                 Normal_loss_module = "CrossEntropyLoss",
                 Intermmediate_loss_module = "MSE",
                 no_students = 4,
                 no_blocks = 3,
                 T = 3,
                 alpha = 0.4,         # Contribution ratio for Normal label loss
                 beta = 0.3,          # Contribution ratio for Intermmediate loss
                 gamma = 0.3          # Contribution ratio for KL Loss 
                 ):

        # Loss module for combined loss computation from three individual losses

        super(Combined_Loss,self).__init__()

        self.Normal_Loss = Normal_Loss(loss_module=Normal_loss_module)
        self.Intermmediate_loss = Intermmediate_loss(no_students=no_students,
                                                     no_blocks= no_blocks,
                                                     Loss_module=Intermmediate_loss_module
                                                     )
        self.KL_Loss = KL_Loss(T=T) 
        self.contri_params = [alpha,beta,gamma]
        self.Individual_loss = []                                           


    def forward(self,preds,labels,intermmediate_maps):

        # preds shape              : [preds of teacher model, preds of Student model 1]
        # individual preds shape   : (batch_size, no_classes)
        # Labels shape             : (Batch_size, no_classes)
        # intermmediate_maps shape : [list of maps per block level 1 ,list of maps per block level 2, .... ]
        # Each map shape           : (Batch_size,no_channels,width,height)

        Combined_loss = 0
        self.Individual_loss = []

        # Normal loss computation 

        Individual_normal_losses = self.Normal_Loss(preds,labels)

        Loss_A = self.contri_params[0]*sum(Individual_normal_losses)
        self.Individual_loss.append(Loss_A.item())
        Combined_loss += Loss_A

        # Intermmediate loss computation 

        Loss_B = self.contri_params[1]*self.Intermmediate_loss(intermmediate_maps)
        self.Individual_loss.append(Loss_B.item())
        Combined_loss += Loss_B

        # KL loss computation 

        teacher_pred = preds[0]
        Loss_C = 0

        for i in range(1,len(preds)):
            Student_pred = preds[i]
            Loss_C += self.contri_params[2]*self.KL_Loss(teacher_pred,Student_pred)

        Combined_loss +=Loss_C
        self.Individual_loss.append(Loss_C.item())

        return Combined_loss,Individual_normal_losses