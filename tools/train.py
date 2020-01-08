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
    Current_cfg = dict()

    # Prepare optimizer
    optim_cfg = Train_cfg['optimizer']
    optim_name = optim_cfg.pop('name')
    print('### SELECTED OPTIMIZER:', optim_name)
    optim_cls = getattr(optim, optim_name)
    optimizer = optim_cls(model.parameters(), **optim_cfg)
    Current_cfg["Optimizer"] = optimizer

    # Prepare loss function(s)
    loss_cfg = Train_cfg.pop('criterion')
    L1_loss_name = loss_cfg.pop('L1')
    L2_loss_name = loss_cfg.pop('L2')
    L3_loss_name = loss_cfg.pop('L3')

    no_students = Model_cfg["No_students"]
    no_blocks = Model_cfg["No_blocks"]
    Temp = Train_cfg["KL_loss_temperature"]
    Contribution_ratios = Train_cfg["Loss_contribution"]

    assert L2_loss_name == "KL_Loss", "Only KL loss is supported for probability distribution comparision"

    print('### SELECTED LOSS FUNCTION:\nL1:{0}\nL2:{1}\nL3:{2}\n'.format(L1_loss_name,L2_loss_name,L3_loss_name))

    # Intializing overall loss computation

    loss = Total_loss(Normal_loss_module = L1_loss_name,
                 Intermmediate_loss_module = L3_loss_name,
                 no_students = no_students,
                 no_blocks = no_blocks,
                 T = Temp,
                 alpha = Contribution_ratios["alpha"],        
                 beta = Contribution_ratios["beta"],          
                 gamma = Contribution_ratios["gamma"]          
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

    # Getting current run_id

    Current_cfg["Run_id"] = get_run_id()
    Current_cfg["Store_root"] = Train_cfg["training_store_root"]
    Current_cfg["DataParallel"] = configer.model["DataParallel"]

    # Loading Training configs 

    Epochs = Train_cfg["epochs"]
    Test_interval = Train_cfg["test_interval"]
    Current_cfg["Plot_Accuracy"] = Train_cfg["plot_accuracy_graphs"]

    # Setting accuracy and losses lists and constants
    Best_Val_accuracy = 0
    Train_accuracies = [[] for i in range(no_students+1)]
    Train_losses = [[] for i in range(no_students+1)]
    Train_ind_losses = [[] for i in range(3)]
    Val_accuracies = [[] for i in range(no_students+1)]
    Val_losses = [[] for i in range(no_students+1)]
    Val_ind_losses = [[] for i in range(3)]

    print ('---------- Starting Training')
    for i in range(Epochs):

        if (scheduler is not None) and (scheduler_name != 'ReduceLROnPlateau'):

            scheduler.step()

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
                print("Best Validation accuracy found uptil now !! Saving model state....")
                Best_Val_accuracy = Epoch_Val_set_accuracy[0]

                Model_State_Saver(model,configer,Current_cfg,Train_accuracies,Train_losses,Train_ind_losses,Val_accuracies,Val_losses,Val_ind_losses,i)

            if (scheduler is not None) and (scheduler_name == 'ReduceLROnPlateau'):

                scheduler.step(Epoch_Val_set_loss[0])

    Model_State_Saver(model,configer,Current_cfg,Train_accuracies,Train_losses,Train_ind_losses,Val_accuracies,Val_losses,Val_ind_losses,i)


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
    running_individual_losses = np.zeros(3)
    Total_correct = np.zeros(No_students+1)

    Total_count = 0

    print('*' * 20, 'TRAINING EPOCH {}'.format(i+1), '*' * 20)

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
            running_student_losses[j]+= Individual_normal_losses[j].item()
        

        Correct_batch = np.asarray(get_count(outputs, labels))
        Total_correct += Correct_batch

        Total_count += len(Input)

        print('Iter: {}/{} | Running loss: Overall : {:.3f} '
        ' Individual : Normal:{:.3f} Intermmediate:{:.3f} KL:{:.3f} Student : {} | Time elapsed: {:.2f} mins'.format(batch_idx + 1, num_train_batches,
                                running_combined_loss/(batch_idx + 1),
                                running_individual_losses[0]/(batch_idx + 1),
                                running_individual_losses[1]/(batch_idx + 1),
                                running_individual_losses[2]/(batch_idx + 1),
                                list(running_student_losses/(batch_idx + 1)),
                                (time.time() - start) / 60), end='\r',flush=True)
        sys.stdout.flush()

        del Input, labels, outputs, Intermmediate_maps

    Epoch_avg_loss = float(running_combined_loss)/num_train_batches
    Epoch_loss_list = [Epoch_avg_loss] + list(running_student_losses/num_train_batches)
    Epoch_accuracy = list((Total_correct/Total_count)*100)
    Epoch_individual_loss_list = list(running_individual_losses/num_train_batches)

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
    running_individual_losses = np.zeros(3)
    Total_correct = np.zeros(No_students+1)

    Total_count = 0

    print('*' * 20, 'VALIDATING EPOCH {}'.format(i+1), '*' * 20)

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
                running_student_losses[j]+= Individual_normal_losses[j].item()

            Correct_batch = np.asarray(get_count(outputs, labels))
            Total_correct += Correct_batch

            Total_count += len(Input)

            print('Iter: {}/{} | Running loss: Overall : {:.3f} '
            ' Individual : Normal:{:.3f} Intermmediate:{:.3f} KL:{:.3f} Student : {} | Time elapsed: {:.2f} mins'.format(batch_idx + 1, num_train_batches,
                                    running_combined_loss/(batch_idx + 1),
                                    running_individual_losses[0]/(batch_idx + 1),
                                    running_individual_losses[1]/(batch_idx + 1),
                                    running_individual_losses[2]/(batch_idx + 1),
                                    list(running_student_losses/(batch_idx + 1)),
                                    (time.time() - start) / 60), end='\r',flush=True)
            sys.stdout.flush()

            del Input, labels, outputs, Intermmediate_maps

    Epoch_avg_loss = float(running_combined_loss)/num_train_batches
    Epoch_loss_list = [Epoch_avg_loss] +  list(running_student_losses/num_train_batches)
    Epoch_accuracy = list((Total_correct/Total_count)*100)
    Epoch_individual_loss_list = list(running_individual_losses/num_train_batches)

    print('\nValidating --> \nAccuracy: Teacher: {:.3f}\nStudents: {} |\nOverall Loss: {:.3f}'.format(Epoch_accuracy[0],  
                                                                                                          Epoch_accuracy[1:],
                                                                                                          Epoch_avg_loss
                                                                                                          ))

    return model,Epoch_accuracy,Epoch_loss_list,Epoch_individual_loss_list


def Model_State_Saver(model,
                      configer,
                      Current_cfg,
                      Train_accuracies,
                      Train_losses,
                      Train_ind_losses,
                      Val_accuracies,
                      Val_losses,
                      Val_ind_losses,
                      i
                      ):

    Store_root = Current_cfg["Store_root"]
    run_id = Current_cfg["Run_id"]
    No_students = configer.model["No_students"]
    plot_accuracy = Current_cfg["Plot_Accuracy"]

    if not os.path.isdir(os.path.join(Store_root,run_id)):
        os.mkdir(os.path.join(Store_root,run_id))
        os.mkdir(os.path.join(Store_root,run_id,'Model_saved_states'))

        os.mkdir(os.path.join(Store_root,run_id,"Plots"))

        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays"))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training'))
        os.mkdir(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation'))

        shutil.copy('../'+args.cfg , os.path.join(Store_root,run_id,"Train_config.py"))

    # Saving model state dict
    if Current_cfg["DataParallel"]:
        torch.save(model.module.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}.pth".format(i)))
    else:
        torch.save(model.state_dict(),os.path.join(Store_root,run_id,'Model_saved_states',"Epoch_{}.pth".format(i)))

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
            plot_accuracy = plot_accuracy
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
    count_parameters(model)

    # Resuming training from saved checkpoint
    if configer.Train_resume:

        Store_root = configer.train_cfg["training_store_root"]
        Load_run_id = configer.Load_run_id
        Load_Epoch = configer.Load_Epoch

        print("\n### Resuming training from config checkpoint ID {0} and Epoch {1}\n".format(Load_run_id,Load_Epoch))

        checkpoint_weights = torch.load(os.path.join(Store_root,Load_run_id,"Model_saved_states","Epoch_{}.pth".format(Load_Epoch)))

        if configer.model["DataParallel"]:
            model.module.load_state_dict(checkpoint_weights)        
        else:
            model.load_state_dict(checkpoint_weights)  

    elif configer.Validate_only:

        Store_root = configer.train_cfg["training_store_root"]
        Load_run_id = configer.Load_run_id
        Load_Epoch = configer.Load_Epoch
        model_name = configer.model["name"]

        print("\n### Validating Model:{0} from config checkpoint ID {1} and Epoch {2}\n".format(model_name,Load_run_id,Load_Epoch))

        Train_cfg = configer.train_cfg
        loss_cfg = Train_cfg['criterion']
        loss_name = loss_cfg.pop('L1')
        loss_cls = getattr(nn, loss_name)
        loss = loss_cls(**loss_cfg)
        Val_cfg = dict(Loss_criterion= loss)

        Val_epoch(configer,model,Val_loader,Val_cfg,0)      

        return 0  

    # Training the model for config settings

    trainer(configer,model,Train_loader,Val_loader)


if __name__ == "__main__":

    args = parser()
    main(args)
