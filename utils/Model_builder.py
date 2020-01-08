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
    gpu_ids = configer.model["gpu"]
    Base_freeze = configer.model["Common_base_freeze"]
    no_students = configer.model["No_students"]
    no_blocks = configer.model["No_blocks"]

    if "Resnet" in model_name:
        
        Resnet_model = getattr(Resnet,"BIO_"+model_name)
        model = Resnet_model(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze,
                             no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,gpus=gpu_ids)

    else:
        raise ImportError("Model Architecture not supported")


    model = model.to(device)

    print ('---------- Model Loaded')
    
    return model