import torch 
import torch.nn as nn
import sys
sys.path.insert(0, '../')

from models import Resnet,Densenet,EfficientNet,ResNet_cifar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Model_builder(configer):

    model_name = configer.model['name']
    No_students = configer.model["No_students"]
    No_blocks = configer.model["No_blocks"]
    No_classes = configer.dataset_cfg["id_cfg"]["num_classes"]
    model_pretrained = configer.model['pretrained']
    model_dataparallel = configer.model["DataParallel"]
    model_gpu_replica = configer.model["Multi_GPU_replica"]
    gpu_ids = configer.model["gpu"]
    Base_freeze = configer.model["Common_base_freeze"]
    Single_model = configer.Single_model_mode
    Common_Base = configer.model["Collective_Base_gradient"]
    no_students = configer.model["No_students"]
    no_blocks = configer.model["No_blocks"]

    if "Wide_Resnet" in model_name:
        
        Resnet_model = getattr(Resnet,"BIO_"+model_name)
        model = Resnet_model(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze,
                             no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,
                             gpus=gpu_ids,Common_Base=Common_Base,Single_model = Single_model)

    elif "Resnet" in model_name:
        
        Resnet_model = getattr(Resnet,"BIO_"+model_name)
        model = Resnet_model(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze,
                             no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,
                             gpus=gpu_ids,Common_Base=Common_Base,Single_model = Single_model)

    elif "ResNet" in model_name:
        
        Resnet_model = getattr(ResNet_cifar,"BIO_"+model_name)
        model = Resnet_model(num_classes = No_classes,Base_freeze=Base_freeze,
                             no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,
                             gpus=gpu_ids,Common_Base=Common_Base,Single_model = Single_model)

    elif "Resnext" in model_name:
        
        Resnet_model = getattr(Resnet,"BIO_"+model_name)
        model = Resnet_model(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze,
                             no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,
                             gpus=gpu_ids,Common_Base=Common_Base,Single_model = Single_model)

    elif "Densenet" in model_name:
        Densenet_model = getattr(Densenet,"BIO_"+model_name)

        model = Densenet_model(num_classes = No_classes,Base_freeze=Base_freeze,
                               no_students=no_students,no_blocks=no_blocks,
                               parallel=model_dataparallel,gpus=gpu_ids,Common_Base=Common_Base,
                               Single_model = Single_model)

    elif "Efficientnet" in model_name:
        Densenet_model = getattr(EfficientNet,"BIO_"+model_name)

        model = Densenet_model(num_classes = No_classes,pretrained=model_pretrained,Base_freeze=Base_freeze,
                               no_students=no_students,no_blocks=no_blocks,parallel=model_dataparallel,
                               gpus=gpu_ids,Common_Base=Common_Base,Single_model = Single_model)

    else:
        raise ImportError("Model Architecture not supported")


    model = model.to(device)

    print ('\n---------- Model Loaded')
    print("Model Architecture: {}\n".format(model_name))

    return model