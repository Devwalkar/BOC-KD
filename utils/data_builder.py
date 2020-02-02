import torchvision.datasets as Data 
import torchvision.transforms as transforms
import torch.utils.data as TD 
import torch
import os 
from .Caltech_loader import Caltech256 
from .ImageNet_loader import ImageNetDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Dataset_Loader(configer):

    # This helper function loads the config dataset and created the torch Dataloader

    data_configer = configer.dataset_cfg
    Dataset_name = data_configer['id_cfg']['name']
    Data_root = data_configer['id_cfg']['root']
    Data_download = data_configer['id_cfg']['download']

    Model = configer.model["name"] 

    ####### Image Transforms builder

    if Dataset_name == "CIFAR10":

        img_transform  = transforms.Compose([transforms.Pad(4),
                                            #transforms.RandomAffine((-20,20)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomCrop((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
                                            ])

        test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])

    elif Dataset_name == "Imagenet":
        img_transform  = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.Pad(50),
                                            transforms.RandomAffine((-10,10)),
                                            transforms.RandomCrop((224,224)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                            ])
 
        test_transform = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
         
    else:
        raise ImportError("DL model architecture not supported for Img transforms")   



    ####### Dataset train and test builder

    Train_configer = data_configer['train_cfg']
    Val_configer = data_configer['val_cfg']

    if Dataset_name == "MNIST":   # Shape: (1,28,28)

        if Data_download:

            Trainloader = Data.MNIST(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.MNIST(Data_root,download=True,train=False, transform = test_transform)
        else:

            Trainloader = Data.MNIST(os.path.join(Data_root,"MNIST"),download=False,train=True,transform = img_transform)
            Testloader = Data.MNIST(os.path.join(Data_root,"MNIST"),download=False,train=False, transform = test_transform)

    elif Dataset_name == "CIFAR10":  # Shape: (3,32,32)

        if Data_download:

            if not os.path.isdir(os.path.join(Data_root,"cifar10")):
                os.mkdir(os.path.join(Data_root,"cifar10")) 
         
            Trainloader = Data.CIFAR10(os.path.join(Data_root,"cifar10"),download=True,train=True,transform = img_transform)
            Testloader = Data.CIFAR10(os.path.join(Data_root,"cifar10"),download=True,train=False, transform = test_transform)

        else:

            Trainloader = Data.CIFAR10(os.path.join(Data_root,"cifar10"),download=False,train=True,transform = img_transform)
            Testloader = Data.CIFAR10(os.path.join(Data_root,"cifar10"),download=False,train=False, transform = test_transform)

    elif Dataset_name == "CIFAR100":  # Shape: (3,32,32)

        if Data_download:
         
            Trainloader = Data.CIFAR100(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.CIFAR100(Data_root,download=True,train=False, transform = test_transform)
        
        else:

            Trainloader = Data.CIFAR100(Data_root,download=False,train=True,transform = img_transform)
            Testloader = Data.CIFAR100(Data_root,download=False,train=False, transform = test_transform)

    elif Dataset_name == "Fashion-MNIST":

        if Data_download:

            Trainloader = Data.FashionMNIST(Data_root,download=True,train=True,transform = img_transform)
            Testloader = Data.FashionMNIST(Data_root,download=True,train=False, transform = test_transform)

        else:           
         
            Trainloader = Data.FashionMNIST(os.path.join(Data_root,"Fashion-MNIST"),download=False,train=True,transform = img_transform)
            Testloader = Data.FashionMNIST(os.path.join(Data_root,"Fashion-MNIST"),download=False,train=False, transform = test_transform)

    elif Dataset_name == "SVHN":

        if Data_download:

            if not os.path.isdir(os.path.join(Data_root,"SVHN")):
                os.mkdir(os.path.join(Data_root,"SVHN")) 
             
            Trainloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=True,split="train",transform = img_transform)
            Testloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=True,split="test", transform = test_transform)
        
        else:

            Trainloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=False,split="train",transform = img_transform)
            Testloader = Data.SVHN(os.path.join(Data_root,"SVHN"),download=False,split="test", transform = test_transform)

    elif Dataset_name == "STL10":

        if Data_download:

            Trainloader = Data.STL10(os.path.join(Data_root),download=True,split="train",transform = img_transform)
            Testloader = Data.STL10(os.path.join(Data_root),download=True,split="test", transform = test_transform)

        else:

            Trainloader = Data.STL10(os.path.join(Data_root),download=False,split="train",transform = img_transform)
            Testloader = Data.STL10(os.path.join(Data_root),download=False,split="test", transform = test_transform)

    elif Dataset_name == "Caltech":

        if Data_download:

            if not os.path.isdir(os.path.join(Data_root,"Caltech")):
                os.mkdir(os.path.join(Data_root,"Caltech")) 

            Trainloader = Caltech256(os.path.join(Data_root,"Caltech"),download=True,train=True,transform=img_transform)
            Testloader = Caltech256(os.path.join(Data_root,"Caltech"),download=True,train=False, transform = test_transform)

        else:

            Trainloader = Caltech256(os.path.join(Data_root,"Caltech"),train=True,transform=img_transform)
            Testloader = Caltech256(os.path.join(Data_root,"Caltech"),train=False, transform = test_transform)

    elif Dataset_name == "Imagenet":

        Trainloader = ImageNetDataset(os.path.join(Data_root,"ImageNet/ILSVRC-train.lmdb"),transform=img_transform)
        Testloader = ImageNetDataset(os.path.join(Data_root,"ImageNet/ILSVRC-val.lmdb"),transform=test_transform)

    else:
        raise ImportError("Dataset not supported")   


    # Creating train and test loaders if not ImageNet dataset

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
    print("Dataset: {}".format(Dataset_name))

    return train_loader,test_loader