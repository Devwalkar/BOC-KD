import torch 
import pretrainedmodels as PM 
import torch.nn as nn
import sys
sys.path.insert(0, '../')

from models import Resnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Model_builder(configer):

    model_name = configer.model['name']
    No_classes = configer.dataset_cfg["id_cfg"]["num_classes"]
    model_pretrained = configer.model['pretrained']
    model_dataparallel = configer.model["DataParallel"]
    model_gpu_replica = configer.model["Multi_GPU_replica"]
    gpu_ids = configer.train_cfg["gpu"]
    Base_freeze = configer.model["Common_base_freeze"]

    if model_name == "Resnet18":
        model = Resnet.BIO_Resnet18(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze)

    elif model_name == "Resnet34":
        model = Resnet.BIO_Resnet34(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze)

    elif model_name == "Resnet50":
        model = Resnet.BIO_Resnet50(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze)

    elif model_name == "Resnet101":
        model = Resnet.BIO_Resnet101(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze)

    elif model_name == "Resnet152":
        model = Resnet.BIO_Resnet152(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze)

    else:
        raise ImportError("Model Architecture not supported")

    # Performing Data Parallelism if configured

    if model_dataparallel:      

        model = torch.nn.DataParallel(model.to(device),device_ids =gpu_ids) 

    elif model_gpu_replica:

        torch.distributed.init_process_group(backend='nccl',world_size=1,rank=1)
        model = torch.nn.DistributedDataParallel(model.to(device),device_ids =gpu_ids)       

    else:
        model = model.to(device)

    print ('---------- Model Loaded')
    
    return model