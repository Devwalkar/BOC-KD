import torch
import torchvision  
import torch.nn as nn 
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data 
import torch.optim as optim
from datetime import datetime
import sys
sys.path.insert(0, '../')

import os 
import shutil 
import torchvision.datasets as Data 
import torch.utils.data as TD
import matplotlib.pyplot as plt 
import time 
from os import path 
from importlib import import_module
from utils.data_builder import Dataset_Loader
from utils.Model_builder import Model_builder
from utils.Losses import Combined_Loss as Total_loss
from utils.Accuracy_Saver_and_Plotter import saver_and_plotter

import pretrainedmodels as PM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parser():
    parser = argparse.ArgumentParser(description='Parser for training')
    parser.add_argument('--cfg',default="configs/config_template.py",help='Configuration file')
    args = parser.parse_args()
    return args

def count_parameters(model):

    total = 0
    trainable = 0
    for param in model.parameters():
        temp = param.numel()
        total += temp
        if param.requires_grad:
            trainable += temp
    print('Total params: {} | Trainable params: {}'.format(
        total, trainable
    ))

def get_run_id():
    """A run ID for a particular training session.
    """
    dt = datetime.now()
    run_id = dt.strftime('%m_%d_%H_%M')
    return run_id

def trainer(configer,model,Train_loader,Val_loader):

    Train_cfg = configer.train_cfg
    Model_cfg = configer.model
    Store_root = configer.train_cfg["training_store_root"]
    no_students = Model_cfg["No_students"]
    Current_cfg = dict()

    # Prepare optimizer
    optim_cfg = Train_cfg['optimizer']
    optim_name = optim_cfg.pop('name')
    print('### SELECTED OPTIMIZER:', optim_name)
    optim_cls = getattr(optim, optim_name)
    
    param_list = [{'params':model.BaseNet.parameters()}]
    for i in range(no_students):
        param_list.append({'params':model.student_models[i].parameters()})

    optimizer_pretrain = optim_cls(param_list[:2], **optim_cfg)
    if configer.Single_model_mode is not None:
        assert configer.Single_model_mode <= (no_students-1),"Single model number should be at max one less than no of students"

        optimizer_student_baseline = optim_cls([param_list[0]]+[param_list[int(configer.Single_model_mode)+1]], **optim_cfg)

    optimizer = optim_cls(param_list, **optim_cfg)

    Current_cfg["Optimizer"] = optimizer
    Current_cfg["Optimizer_pretrain"] = optimizer_pretrain
    Current_cfg["No_students"] = no_students

    # Prepare loss function(s)
    loss_cfg = Train_cfg.pop('criterion')
    L1_loss_name = loss_cfg['L1']
    L2_loss_name = loss_cfg.pop('L2')
    L3_loss_name = loss_cfg.pop('L3')
    Current_cfg["L1"] = L1_loss_name

    no_blocks = Model_cfg["No_blocks"]
    Temp = Train_cfg["KL_loss_temperature"]

    assert L2_loss_name == "KL_Loss", "Only KL loss is supported for probability distribution comparision"

    print('### SELECTED LOSS FUNCTION:\nL1:{0}\nL2:{1}\nL3:{2}\n'.format(L1_loss_name,L2_loss_name,L3_loss_name))

    # Intializing overall loss computation

    loss = Total_loss(pretrain_mode=False,
                 Normal_loss_module = L1_loss_name,
                 Intermmediate_loss_module = L3_loss_name,
                 no_students = no_students,
                 no_blocks = no_blocks,
                 T = Temp         
                 )

    Current_cfg["Loss_criterion"] = loss

    # Prepare learning rate scheduler if specified
    scheduler_cfg = Train_cfg.pop('scheduler', None)
    scheduler = None


    if scheduler_cfg is not None:
        scheduler_name = scheduler_cfg.pop('name')
        print('### SELECTED SCHEDULER:{}\n'.format(scheduler_name))
        scheduler_cls = getattr(optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_cls(optimizer, **scheduler_cfg)

    Current_cfg["scheduler"] = scheduler
    Current_cfg["scheduler_name"] = scheduler_name

    # Resume training configs
    Resume = configer.Train_resume
    Load_run_id = configer.Load_run_id
    Load_epoch = configer.Load_Epoch
    Current_cfg["Resume"] = Resume

    # Getting current run_id

    Current_cfg["Run_id"] = Load_run_id if Resume else get_run_id() 
    Current_cfg["Store_root"] = Store_root
    Current_cfg["DataParallel"] = configer.model["DataParallel"]
    Current_cfg["Dataset"]  = configer.dataset_cfg['id_cfg']['name']

    # Loading Training configs 

    Epochs = Train_cfg["epochs"]
    Test_interval = Train_cfg["test_interval"]
    Current_cfg["Plot_Accuracy"] = Train_cfg["plot_accuracy_graphs"]
    Current_cfg["pretrain_epochs"] = Train_cfg["pretraining_epochs"]
    Current_cfg["test_interval"] = Test_interval
    Current_cfg["Baseline_Epochs"] = Epochs

    # Single model baseline training
    Single_model = configer.Single_model_mode
    Current_cfg["Load_epoch"] = Load_epoch

    # Teacher pretraining stage
    teacher_pretraining = Train_cfg["teacher_pretraining"]

    if Single_model is not None:

        Current_cfg["Baseline_optimizer"] = optimizer_student_baseline
        Current_cfg["Student_model"] = Single_model
        model.Single_mode()
        Single_Model_training(Current_cfg,model,Train_loader,Val_loader)
        return None

    elif teacher_pretraining:
        model.pretrain()
        model = Model_Pretraining(Current_cfg,model,Train_loader,Val_loader)
        model.student_version()

    # Setting accuracy and losses lists and constants
    Best_Val_accuracy = 0

    if Resume:
        load_path = os.path.join(Store_root,Load_run_id,"Accuracy_arrays")

        Train_accuracies = [list(np.load(os.path.join(load_path,"Training","Train_Accuracy_for_Combined_Teacher.npy"),encoding="bytes"))]
        Train_accuracies += [list(np.load(os.path.join(load_path,"Training","Train_Accuracy_for_Student_{}.npy".format(i+1)),encoding="bytes")) for i in range(no_students)]        
        Train_losses = [list(np.load(os.path.join(load_path,"Training","Train_loss_for_Overall_Loss.npy"),encoding="bytes"))]
        Train_losses += [list(np.load(os.path.join(load_path,"Training","Train_loss_for_Student_{}.npy".format(i+1)),encoding="bytes")) for i in range(no_students)]

        Val_accuracies = [list(np.load(os.path.join(load_path,"Validation","Valid_Accuracies_for_Combined_Teacher.npy"),encoding="bytes"))]
        Val_accuracies += [list(np.load(os.path.join(load_path,"Validation","Valid_Accuracies_for_Student_{}.npy".format(i+1)),encoding="bytes")) for i in range(no_students)]        
        Val_losses = [list(np.load(os.path.join(load_path,"Validation","Val_loss_for_Overall_Loss.npy"),encoding="bytes"))]
        Val_losses += [list(np.load(os.path.join(load_path,"Validation","Val_loss_for_Student_{}.npy".format(i+1)),encoding="bytes")) for i in range(no_students)]
     
    else:
        Train_accuracies = [[] for i in range(no_students+1)]
        Train_losses = [[] for i in range(no_students+1)]
        Val_accuracies = [[] for i in range(no_students+1)]
        Val_losses = [[] for i in range(no_students+1)]

    Train_ind_losses = [[] for i in range(3)]
    Val_ind_losses = [[] for i in range(3)]

    if Resume:
        print ('---------- Resuming Training')
        start_epoch = Load_epoch  
    else:     
        print ('---------- Starting Training')
        start_epoch =  0

    for i in range(start_epoch,start_epoch+Epochs):

        model,Epoch_train_set_accuracy,Epoch_train_set_loss,Epoch_train_individual_loss = Train_epoch(configer,model,Train_loader,Current_cfg,i)

        for j in range(no_students+1):
            Train_accuracies[j].append(Epoch_train_set_accuracy[j])
            Train_losses[j].append(Epoch_train_set_loss[j])
        
        for k in range(3):
            Train_ind_losses[k].append(Epoch_train_individual_loss[k])

        if (i%Test_interval) == 0:

            model,Epoch_Val_set_accuracy,Epoch_Val_set_loss,Epoch_Val_individual_loss = Val_epoch(configer,model,Val_loader,Current_cfg,i)           

            for j in range(no_students+1):
                Val_accuracies[j].append(Epoch_Val_set_accuracy[j])
                Val_losses[j].append(Epoch_Val_set_loss[j])
            
            for k in range(3):
                Val_ind_losses[k].append(Epoch_Val_individual_loss[k])

            if Epoch_Val_set_accuracy[0] > Best_Val_accuracy:
                print("Best Validation accuracy found uptil now !!")
                Best_Val_accuracy = Epoch_Val_set_accuracy[0]

            print("----- Saving model state")
            Model_State_Saver(model,configer,Current_cfg,Train_accuracies,Train_losses,Train_ind_losses,Val_accuracies,Val_losses,Val_ind_losses,i)

            if (scheduler is not None) and (scheduler_name == 'ReduceLROnPlateau'):

                scheduler.step(Epoch_Val_set_accuracy[0])
            
            elif scheduler_name == "MultiStepLR":
                
                scheduler.step()

    Model_State_Saver(model,configer,Current_cfg,Train_accuracies,Train_losses,Train_ind_losses,Val_accuracies,Val_losses,Val_ind_losses,i)


def Model_Pretraining(Current_cfg,model,Train_loader,Val_loader):

    def get_count(outputs, labels):
        
        #Number of correctly predicted labels for each student model.       
        pred_labels = torch.argmax(outputs, dim=1)
        count       = torch.sum(torch.eq(pred_labels, labels)).item()
        return count

    # Training data

    Epochs = Current_cfg["pretrain_epochs"]
    optimizer = Current_cfg['Optimizer_pretrain']
    Dataset = Current_cfg["Dataset"] 
    scheduler = Current_cfg["scheduler"]
    scheduler_name = Current_cfg["scheduler_name"]
    Test_interval = Current_cfg["test_interval"]

    Best_pretrain_Val_accuracy = 0
    criterion = Total_loss(pretrain_mode=True,
                               Normal_loss_module = Current_cfg["L1"]        
                )

    print ('---------- Starting Psuedo Teacher Model Pre-Training\n')
    for i in range(Epochs):

        print('\n','*' * 20, 'PRE-TRAINING EPOCH {}'.format(i+1), '*'* 20,"\n")

        model.train()
        start = time.time()

        # Some useful constants
        num_train_batches = len(Train_loader)
        num_val_batches = len(Val_loader)
        running_train_loss = 0
        running_val_loss = 0
        Total_train_correct = 0
        Total_val_correct = 0
        Total_train_count = 0
        Total_val_count = 0

        if (scheduler is not None) and (scheduler_name != 'ReduceLROnPlateau'):

            scheduler.step()

        for batch_idx, (Input, labels) in enumerate(Train_loader):

            Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
            Input = Input.float().to(device)
            labels = labels.to(device)

            Batch_output = model(Input)
            Loss = criterion(Batch_output, labels)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            running_train_loss +=Loss.item()
            Total_train_correct += get_count(Batch_output,labels)
            Total_train_count += len(Input)

            print('Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                    ' mins'.format(batch_idx + 1, num_train_batches,
                                    running_train_loss/(batch_idx + 1),
                                    (time.time() - start) / 60), end='\r',
                    flush=True)

        print('\nTraining epoch results --> Accuracy: {:.3f}% | Overall Loss: {:.3f}'.format(float((Total_train_correct)/Total_train_count)*100,  
                                                                                                            float(running_train_loss)/num_train_batches
                                                                                                            ))
            
        if (i%Test_interval) == 0:
            model.eval()
            start = time.time()

            print('\n','*' * 20, 'PRE-VALIDATING EPOCH {}'.format(i+1), '*'* 20,"\n")

            for batch_idx, (Input, labels) in enumerate(Val_loader):
                Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
                Input = Input.float().to(device)
                labels = labels.to(device)

                Batch_output = model(Input)
                Loss = criterion(Batch_output, labels)

                running_val_loss +=Loss.item()
                Total_val_correct += get_count(Batch_output,labels)
                Total_val_count += len(Input)

                print('Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                        ' mins'.format(batch_idx + 1, num_val_batches,
                                        running_val_loss/(batch_idx + 1),
                                        (time.time() - start) / 60), end='\r',
                        flush=True)

            Val_accuracy = (float(Total_val_correct)/Total_val_count)*100
            print('\nValidation epoch results --> Accuracy: {:.3f}% | Overall Loss: {:.3f}'.format(Val_accuracy,  
                                                                                                    float(running_val_loss)/num_val_batches
                                                                                                    ))
            if Val_accuracy > Best_pretrain_Val_accuracy:
                Best_pretrain_Val_accuracy = Val_accuracy
                Model_State_Saver(model,Current_cfg=Current_cfg,i=i)

            if (scheduler is not None) and (scheduler_name == 'ReduceLROnPlateau'):
                scheduler.step(Val_accuracy)
    
    print("\n------------ Pretraining Stage Completed\n")
    return model


def Single_Model_training(Current_cfg,model,Train_loader,Val_loader):

    def get_count(outputs, labels):
        
        #Number of correctly predicted labels for each student model.       
        pred_labels = torch.argmax(outputs, dim=1)
        count       = torch.sum(torch.eq(pred_labels, labels)).item()
        return count

    # Training data

    Epochs = Current_cfg["Baseline_Epochs"]
    optimizer = Current_cfg["Baseline_optimizer"] 
    Dataset = Current_cfg["Dataset"] 
    scheduler = Current_cfg["scheduler"]
    scheduler_name = Current_cfg["scheduler_name"]
    Test_interval = Current_cfg["test_interval"]
    Student_model_name = Current_cfg["Student_model"]
    Resume = Current_cfg["Resume"]
    Load_epoch = Current_cfg["Load_epoch"] 

    Best_Val_accuracy = 0
    criterion = Total_loss(pretrain_mode=True,
                               Normal_loss_module = Current_cfg["L1"]        
                )
    if Resume:
        print ('---------- Resuming Student Model {} Training\n'.format(Student_model_name))   
        start_epoch = Load_epoch
    else:    
        print ('---------- Starting Student Model {} Training\n'.format(Student_model_name))
        start_epoch = 0

    for i in range(start_epoch,(start_epoch+Epochs)):

        print('\n','*' * 20, 'TRAINING EPOCH {}'.format(i+1), '*'* 20,"\n")

        model.train()
        start = time.time()

        # Some useful constants
        num_train_batches = len(Train_loader)
        num_val_batches = len(Val_loader)
        running_train_loss = 0
        running_val_loss = 0
        Total_train_correct = 0
        Total_val_correct = 0
        Total_train_count = 0
        Total_val_count = 0

        if (scheduler is not None) and (scheduler_name != 'ReduceLROnPlateau'):

            scheduler.step()

        for batch_idx, (Input, labels) in enumerate(Train_loader):

            Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
            Input = Input.float().to(device)
            labels = labels.to(device)

            Batch_output = model(Input)
            Loss = criterion(Batch_output, labels)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            running_train_loss +=Loss.item()
            Total_train_correct += get_count(Batch_output,labels)
            Total_train_count += len(Input)

            print('Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                    ' mins'.format(batch_idx + 1, num_train_batches,
                                    running_train_loss/(batch_idx + 1),
                                    (time.time() - start) / 60), end='\r',
                    flush=True)

        print('\nTraining epoch results --> Accuracy: {:.3f}% | Overall Loss: {:.3f}'.format(float((Total_train_correct)/Total_train_count)*100,  
                                                                                                            float(running_train_loss)/num_train_batches
                                                                                                            ))
            
        if (i%Test_interval) == 0:
            model.eval()
            start = time.time()

            print('\n','*' * 20, 'VALIDATING EPOCH {}'.format(i+1), '*'* 20,"\n")

            for batch_idx, (Input, labels) in enumerate(Val_loader):
                Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
                Input = Input.float().to(device)
                labels = labels.to(device)

                Batch_output = model(Input)
                Loss = criterion(Batch_output, labels)

                running_val_loss +=Loss.item()
                Total_val_correct += get_count(Batch_output,labels)
                Total_val_count += len(Input)

                print('Iter: {}/{} | Running loss: {:.3f} | Time elapsed: {:.2f}'
                        ' mins'.format(batch_idx + 1, num_val_batches,
                                        running_val_loss/(batch_idx + 1),
                                        (time.time() - start) / 60), end='\r',
                        flush=True)

            Val_accuracy = (float(Total_val_correct)/Total_val_count)*100
            print('\nValidation epoch results --> Accuracy: {:.3f}% | Overall Loss: {:.3f}'.format(Val_accuracy,  
                                                                                                    float(running_val_loss)/num_val_batches
                                                                                                    ))
            if Val_accuracy > Best_Val_accuracy:
                Best_Val_accuracy = Val_accuracy
                Model_State_Saver(model,Current_cfg=Current_cfg,i=i,Single_model_mode=True)

            if (scheduler is not None) and (scheduler_name == 'ReduceLROnPlateau'):
                scheduler.step(Val_accuracy)
    
    print("\n------------ Student training Stage Completed\n")


def Train_epoch(configer,model,Train_loader,Current_cfg,i):

    def get_count(outputs, labels):
        
        #Number of correctly predicted labels for each student model.
        
        count_list = []
        for single_model_output in outputs:
            pred_labels = torch.argmax(single_model_output, dim=1)
            count_list.append(torch.sum(torch.eq(pred_labels, labels)).item())
        return count_list

    # Training essentials

    optimizer = Current_cfg['Optimizer']
    criterion = Current_cfg['Loss_criterion']
    Dataset = configer.dataset_cfg['id_cfg']['name']
    No_students = configer.model["No_students"]

    # Some useful constants
    num_train_batches = len(Train_loader)
    running_combined_loss = 0
    running_student_losses = np.zeros(No_students)
    running_teacher_losses = 0
    running_individual_losses = np.zeros(3)
    Total_correct = np.zeros(No_students+1)

    Total_count = 0

    print('\n','*' * 20, 'TRAINING EPOCH {}'.format(i+1), '*'* 20,"\n")

    start = time.time()
    model.train()

    for batch_idx, (Input, labels) in enumerate(Train_loader):

        Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
        Input = Input.float().to(device)
        labels = labels.to(device)

        outputs,Intermmediate_maps = model(Input)
        Combined_loss,Individual_normal_losses = criterion(outputs, labels, Intermmediate_maps)

        optimizer.zero_grad()
        Combined_loss.backward()
        optimizer.step()

        running_combined_loss += Combined_loss.item()
        running_individual_losses += np.asarray(criterion.Individual_loss)

        for j in range(No_students):
            running_student_losses[j]+= Individual_normal_losses[j+1].item()

        running_teacher_losses += Individual_normal_losses[0].item()       
        Correct_batch = np.asarray(get_count(outputs, labels))
        Total_correct += Correct_batch

        Total_count += len(Input)

        print('Iter: {}/{} | Running loss: Overall : {:.3f} '
        ' Individual : Normal:{:.3f} Intermmediate:{:.3f} KL:{:.3f} Teacher : {:.3f} | Time elapsed: {:.2f} mins'.format(batch_idx + 1, num_train_batches,
                                running_combined_loss/(batch_idx + 1),
                                running_individual_losses[0]/(batch_idx + 1),
                                running_individual_losses[1]/(batch_idx + 1),
                                running_individual_losses[2]/(batch_idx + 1),
                                running_teacher_losses/(batch_idx + 1),
                                (time.time() - start) / 60), end='\r',flush=True)
        sys.stdout.flush()

        del Input, labels, outputs, Intermmediate_maps

    Epoch_avg_loss = float(running_combined_loss)/num_train_batches
    Epoch_loss_list = [Epoch_avg_loss] + list(running_student_losses/num_train_batches)
    Epoch_accuracy = list((Total_correct/Total_count)*100)
    Epoch_individual_loss_list = list(running_individual_losses/num_train_batches)

    print('\ntraining epoch results --> \nAccuracy: Teacher: {:.3f}\nStudents: {} |\nOverall Loss: {:.3f}'.format(Epoch_accuracy[0],  
                                                                                                          Epoch_accuracy[1:],
                                                                                                          Epoch_avg_loss
                                                                                                          ))

    return model,Epoch_accuracy,Epoch_loss_list,Epoch_individual_loss_list


def Val_epoch(configer,model,Val_loader,Current_cfg,i):  

    def get_count(outputs, labels):
        
        #Number of correctly predicted labels for each student model.
        
        count_list = []
        for single_model_output in outputs:
            pred_labels = torch.argmax(single_model_output, dim=1)
            count_list.append(torch.sum(torch.eq(pred_labels, labels)).item())
        return count_list

    # Training essentials

    criterion = Current_cfg['Loss_criterion']
    Dataset = configer.dataset_cfg['id_cfg']['name']
    No_students = configer.model["No_students"]

    # Some useful constants
    num_train_batches = len(Val_loader)
    running_combined_loss = 0
    running_student_losses = np.zeros(No_students)
    running_teacher_losses = 0
    running_individual_losses = np.zeros(3)
    Total_correct = np.zeros(No_students+1)

    Total_count = 0

    if i is not None:
        print('\n','*' * 20, 'VALIDATING EPOCH {}'.format(i+1), '*' * 20,"\n")

    start = time.time()
    model.eval()

    for batch_idx, (Input, labels) in enumerate(Val_loader):

            Input = Input.repeat(1,3,1,1) if Dataset in ["MNIST", "Fashion-MNIST"] else Input
            Input = Input.float().to(device)
            labels = labels.to(device)

            outputs,Intermmediate_maps = model(Input)
            Combined_loss,Individual_normal_losses = criterion(outputs, labels, Intermmediate_maps)

            running_combined_loss += Combined_loss.item()
            running_individual_losses += np.asarray(criterion.Individual_loss)

            for j in range(No_students):
                running_student_losses[j]+= Individual_normal_losses[j+1].item()

            running_teacher_losses += Individual_normal_losses[0].item()

            Correct_batch = np.asarray(get_count(outputs, labels))
            Total_correct += Correct_batch

            Total_count += len(Input)

            print('Iter: {}/{} | Running loss: Overall : {:.3f} '
            ' Individual: Normal: {:.3f} Intermmediate: {:.3f} KL: {:.3f} Teacher: {:.3f} | Time elapsed: {:.2f} mins'.format(batch_idx + 1, num_train_batches,
                                    running_combined_loss/(batch_idx + 1),
                                    running_individual_losses[0]/(batch_idx + 1),
                                    running_individual_losses[1]/(batch_idx + 1),
                                    running_individual_losses[2]/(batch_idx + 1),
                                    running_teacher_losses/(batch_idx + 1),
                                    (time.time() - start) / 60), end='\r',flush=True)
            sys.stdout.flush()

            del Input, labels, outputs, Intermmediate_maps

    Epoch_avg_loss = float(running_combined_loss)/num_train_batches
    Epoch_loss_list = [Epoch_avg_loss] +  list(running_student_losses/num_train_batches)
    Epoch_accuracy = list((Total_correct/Total_count)*100)
    Epoch_individual_loss_list = list(running_individual_losses/num_train_batches)

    print('\nValidating epoch results--> \nAccuracy: Teacher: {:.3f}\nStudents: {} |\nOverall Loss: {:.3f}'.format(Epoch_accuracy[0],  
                                                                                                          Epoch_accuracy[1:],
                                                                                                          Epoch_avg_loss
                                                                                                          ))

    return model,Epoch_accuracy,Epoch_loss_list,Epoch_individual_loss_list


def Model_State_Saver(model,
                      configer = None,
                      Current_cfg = None,
                      Train_accuracies = None,
                      Train_losses = None,
                      Train_ind_losses = None,
                      Val_accuracies = None,
                      Val_losses = None,
                      Val_ind_losses = None,
                      i = None,
                      Single_model_mode = False,
                      ):

    Store_root = Current_cfg["Store_root"]
    run_id = Current_cfg["Run_id"]
    No_students = Current_cfg["No_students"]
    plot_accuracy = Current_cfg["Plot_Accuracy"]
    Test_interval = Current_cfg["test_interval"]
    Resume = Current_cfg["Resume"]

    if not os.path.isdir(os.path.join(Store_root,run_id)):
        os.mkdir(os.path.join(Store_root,run_id))
        os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states'))

        os.mkdir(os.path.join(Store_root,run_id,"Plots"))

        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays"))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training'))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation'))

        shutil.copy('../'+args.cfg , os.path.join(Store_root,run_id,"Train_config.py"))

        if configer is None:
            if Single_model_mode and (not os.path.isdir(os.path.join(Store_root,run_id,'Model_saved_states',"Student_training"))):
                os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Student_training"))
                os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Student_training","Student_{}".format(Current_cfg["Student_model"])))

            elif not os.path.isdir(os.path.join(Store_root,run_id,'Model_saved_states',"Pretraining")):
                os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Pretraining"))

    if Single_model_mode:

        os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Student_training","Student_{}".format(Current_cfg["Student_model"]),"Epoch_{}".format(i+1)))        
        torch.save(model.BaseNet.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Student_training",
                              "Student_{}".format(Current_cfg["Student_model"]),"Epoch_{}".format(i+1),"BaseNet.pth"))
        torch.save(model.student_models[Current_cfg["Student_model"]].state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',
                         "Student_training","Student_{}".format(Current_cfg["Student_model"]),"Epoch_{}".format(i+1),"Non_BaseNet.pth"))

        return None

    elif configer is None:

        os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Pretraining","Epoch_{}".format(i+1)))
        torch.save(model.BaseNet.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Pretraining","Epoch_{}".format(i+1),"BaseNet.pth"))
        torch.save(model.student_models[0].state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Pretraining","Epoch_{}".format(i+1),"student_0.pth"))

        return None

    os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}".format(i+1))) 
    if Resume:
        shutil.copy('../'+args.cfg , os.path.join(Store_root,run_id,"Train_config_resume.py"))

    DataParallel = configer.model["DataParallel"]

    if DataParallel:
        torch.save(model.BaseNet.module.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}".format(i+1),"BaseNet.pth"))    
        for g in range(No_students):
            torch.save(model.student_models[g].module.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}".format(i+1),"student_{}.pth".format(g)))

    else:
        torch.save(model.BaseNet.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}".format(i+1),"BaseNet.pth"))    
        for g in range(No_students):
            torch.save(model.student_models[g].state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}".format(i+1),"student_{}.pth".format(g)))

    # Saving Accuracy and loss arrays and Plotting Training and Validation plots

    saver_and_plotter(Train_accuracies = Train_accuracies, 
            Val_accuracies = Val_accuracies,
			Train_losses = Train_losses,
            Train_ind_losses = Train_ind_losses,
			Val_losses = Val_losses,
            Val_ind_losses = Val_ind_losses,
			Store_root = Store_root,
			run_id = run_id,
			No_students = No_students,
            plot_accuracy = plot_accuracy,
            Test_interval = Test_interval
			)


def main(args):
    filename = args.cfg
    module_name = path.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = path.dirname(filename)
    sys.path.insert(0, "../"+config_dir)
    configer = import_module(module_name)
    sys.path.pop(0)

    # Building Dataloaders
    Train_loader,Val_loader = Dataset_Loader(configer)

    # Building config DL model
    model = Model_builder(configer)

    # Resuming training from saved checkpoint
    if configer.Train_resume or configer.Validate_only:

        Store_root = configer.train_cfg["training_store_root"]
        Load_run_id = configer.Load_run_id
        Load_Epoch = configer.Load_Epoch
        No_students = configer.model["No_students"]
        model_name = configer.model["name"]
        no_blocks = configer.model["No_blocks"]
        Temp = configer.train_cfg["KL_loss_temperature"]
        DataParallel = configer.model["DataParallel"]
        Single_model_mode = configer.Single_model_mode

        if Single_model_mode is not None:
            Load_path = os.path.join(Store_root,Load_run_id,"Model_saved_states","Student_training","Student_{}".format(Single_model_mode),
                                     "Epoch_{}".format(Load_Epoch))    
        else:        
            Load_path = os.path.join(Store_root,Load_run_id,"Model_saved_states","Epoch_{}".format(Load_Epoch))

        if DataParallel:
            model.BaseNet.module.load_state_dict(torch.load(os.path.join(Load_path,"BaseNet.pth")))  
            if Single_model_mode is not None:
                model.student_models[Single_model_mode].module.load_state_dict(torch.load(os.path.join(Load_path,"Non_BaseNet.pth")))
            else:
                for g in range(No_students):
                    model.student_models[g].module.load_state_dict(torch.load(os.path.join(Load_path,"student_{}.pth".format(g))))
        else:
            model.BaseNet.load_state_dict(torch.load(os.path.join(Load_path,"BaseNet.pth")))  
            if Single_model_mode is not None:
                model.student_models[Single_model_mode].load_state_dict(torch.load(os.path.join(Load_path,"Non_BaseNet.pth")))
            else:
                for g in range(No_students):
                    model.student_models[g].load_state_dict(torch.load(os.path.join(Load_path,"student_{}.pth".format(g))))
        
        print("\n###### Loaded checkpoint ID {} and Epoch {} successfully\n".format(Load_run_id,Load_Epoch))

        if configer.Train_resume:
            print("\n### Resuming training from checkpoint")
            trainer(configer,model,Train_loader,Val_loader)
            return None

        elif configer.Validate_only:

            print("\n### Validating Model\n")

            loss_cfg = configer.train_cfg["criterion"]

            loss = Total_loss(pretrain_mode=False,
                Normal_loss_module = loss_cfg["L1"],
                Intermmediate_loss_module = loss_cfg["L3"],
                no_students = No_students,
                no_blocks = no_blocks,
                T = Temp         
                )

            Val_cfg = dict(Loss_criterion= loss)

            Val_epoch(configer,model,Val_loader,Val_cfg,None)      

            return None  

    # Training the model for config settings

    trainer(configer,model,Train_loader,Val_loader)


if __name__ == "__main__":

    args = parser()
    main(args)
