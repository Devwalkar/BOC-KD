import torchvision.datasets as Data 
import torchvision.transforms as transforms
import torch.utils.data as TD 
import os 
from .Caltech_loader import Caltech256
from .Imagenet_loader import ImageFolder
import utils.transforms as Imagenet_trans

def Dataset_Loader(configer):

    # This helper function loads the config dataset and created the torch Dataloader

    data_configer = configer.dataset_cfg
    Dataset_name = data_configer['id_cfg']['name']
    Data_root = data_configer['id_cfg']['root']
    Data_download = data_configer['id_cfg']['download']

    Model = configer.model["name"] 

    ####### Image Transforms builder

    if Model in ["Inceptionv3","Xception"]:
        if Dataset_name == "Imagenet":
            img_transform  = transforms.Compose([Imagenet_trans.ToTensor(),Imagenet_trans.CenterCrop((299,299))])
        else:
            img_transform  = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])

    elif Model in ["Densenet121","VGG_19","Resnet18","Resnet50","Resnet34",
                        "Resnet101","Resnet152","ResNeXt101-32","ResNeXt101-64"]:

        if Dataset_name == "Imagenet":    
            img_transform  = transforms.Compose([Imagenet_trans.ToTensor(),Imagenet_trans.CenterCrop((224,224))]) 
        elif Dataset_name == "Caltech":
            img_transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            img_transform  = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) 
      
    elif Model in ["MobilenetV2"]:
        if Dataset_name == "Imagenet":
            img_transform = transforms.Compose([
                    Imagenet_trans.ToTensor(),
                    Imagenet_trans.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                    Imagenet_trans.CenterCrop((224,224))])
        else:
            img_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]) 

    else:
        raise ImportError("DL model architecture not supported")   



    ####### Dataset train and test builder

    if Dataset_name == "MNIST":   # Shape: (1,28,28)

        if Data_download:

            Trainloader = Data.MNIST(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.MNIST(Data_root,download=True,train=False,transform = img_transform)
        else:

            Trainloader = Data.MNIST(os.path.join(Data_root,"MNIST"),download=False,train=True,transform = img_transform)
            Testloader = Data.MNIST(os.path.join(Data_root,"MNIST"),download=False,train=False,transform = img_transform)

    elif Dataset_name == "CIFAR10":  # Shape: (3,32,32)

        if Data_download:
         
            Trainloader = Data.CIFAR10(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.CIFAR10(Data_root,download=True,train=False,transform = img_transform)

        else:

            Trainloader = Data.CIFAR10(os.path.join(Data_root),download=False,train=True,transform = img_transform)
            Testloader = Data.CIFAR10(os.path.join(Data_root),download=False,train=False,transform = img_transform)

    elif Dataset_name == "CIFAR100":  # Shape: (3,32,32)

        if Data_download:
         
            Trainloader = Data.CIFAR100(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.CIFAR100(Data_root,download=True,train=False,transform = img_transform)
        
        else:

            Trainloader = Data.CIFAR100(Data_root,download=False,train=True,transform = img_transform)
            Testloader = Data.CIFAR100(Data_root,download=False,train=False,transform = img_transform)

    elif Dataset_name == "Fashion-MNIST":

        if Data_download:

            Trainloader = Data.FashionMNIST(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.FashionMNIST(Data_root,download=True,train=False,transform = img_transform)

        else:           
         
            Trainloader = Data.FashionMNIST(os.path.join(Data_root,"Fashion-MNIST"),download=False,train=True,transform = img_transform)
            Testloader = Data.FashionMNIST(os.path.join(Data_root,"Fashion-MNIST"),download=False,train=False,transform = img_transform)

    elif Dataset_name == "SVHN":

        if Data_download:
             
            Trainloader = Data.SVHN(Data_root,download=True,split="train",transform = img_transform)
            Testloader = Data.SVHN(Data_root,download=True,split="test",transform = img_transform)
        
        else:

            Trainloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=False,split="train",transform = img_transform)
            Testloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=False,split="test",transform = img_transform)

    elif Dataset_name == "STL10":

        if Data_download:

            Trainloader = Data.STL10(os.path.join(Data_root),download=True,split="train",transform = img_transform)
            Testloader = Data.STL10(os.path.join(Data_root),download=True,split="test",transform = img_transform)

        else:

            Trainloader = Data.STL10(os.path.join(Data_root),download=False,split="train",transform = img_transform)
            Testloader = Data.STL10(os.path.join(Data_root),download=False,split="test",transform = img_transform)

    elif Dataset_name == "Caltech":

        if Data_download:

            if not os.path.isdir(os.path.join(Data_root,"Caltech")):
                os.mkdir(os.path.join(Data_root,"Caltech")) 

            Trainloader = Caltech256(os.path.join(Data_root,"Caltech"),download=True,train=True,transform=img_transform)
            Testloader = Caltech256(os.path.join(Data_root,"Caltech"),download=True,train=False,transform=img_transform)

        else:

            Trainloader = Caltech256(os.path.join(Data_root,"Caltech"),train=True,transform=img_transform)
            Testloader = Caltech256(os.path.join(Data_root,"Caltech"),train=False,transform=img_transform)

    elif Dataset_name == "Imagenet":

        Trainloader = ImageFolder(data_path=Data_root,transform=img_transform)
        Testloader = ImageFolder(data_path=Data_root,transform=img_transform,Train=False)

    else:
        raise ImportError("Dataset not supported")   


    # Creating train and test loaders

    Train_configer = data_configer['train_cfg']
    Val_configer = data_configer['val_cfg']

    train_loader = TD.DataLoader(dataset=Trainloader,
                                batch_size= Train_configer['batch_size'],
                                shuffle= Train_configer['shuffle'])
                                #num_workers= Train_configer['num_workers'],
                                #pin_memory=True)

    test_loader = TD.DataLoader(dataset=Testloader,
                                batch_size= Val_configer['batch_size'],
                                shuffle= Val_configer['shuffle'])
                                #num_workers= Val_configer['num_workers'],
                                #pin_memory=True)

    print ('---------- Training and Test data Loaded ')

    return train_loader,test_loader