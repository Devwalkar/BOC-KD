import torch 
import pretrainedmodels as PM 
import torch.nn as nn
from .Mobilenet import MobileNetV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Model_builder(configer):

    model_name = configer.model['name']
    No_classes = configer.dataset_cfg["id_cfg"]["num_classes"]
    model_pretrained = configer.model['pretrained']
    model_dataparallel = configer.model["DataParallel"]
    model_gpu_replica = configer.model["Multi_GPU_replica"]
    gpu_ids = configer.train_cfg["gpu"]

    if model_name == "Inceptionv3":
        model = PM.inceptionv3(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Xception":
        model = PM.xception(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "VGG_19":
        model = PM.vgg19(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Resnet18":
        model = PM.resnet18(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Resnet50":
        model = PM.resnet50(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Resnet101":
        model = PM.resnet101(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Resnet152":
        model = PM.resnet152(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Resnet34":
        model = PM.resnet34(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "Densenet121":
        model = PM.densenet121(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 
 
    elif model_name == "ResNeXt101-32":
        model = PM.resnext101_32x4d(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "ResNeXt101-64":
        model = PM.resnext101_64x4d(num_classes = 1000,pretrained=model_pretrained)
        d = model.last_linear.in_features
        model.last_linear = nn.Linear(d, No_classes) 

    elif model_name == "MobilenetV2":
        model = MobileNetV2(n_class=No_classes)

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