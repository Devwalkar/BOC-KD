""" Template Configuration file 
"""

# DL model Architecture Settings

'''
Choose DL model from  "Resnet20", "Resnet34", "Resnet50", "Resnet101", "Resnet152",
                      "ResNet20","ResNet32","ResNet44","ResNet56","ResNet110","ResNet1202",
                      "Densenet_12_100", "Densenet_24_250", "Densenet_40_190",
                      "Efficientnet_(B0-B7)", "Resnext50_32x4d", "Resnext101_32x8d",
                      "Wide_Resnet50_2", "Wide_Resnet101_2"

'''
model = dict(
        name ="Efficientnet_B6",
        pretrained = False,           # Select between True and False
        No_students = 5,              # Number of student models to create for training
        No_blocks = 3,                # Number of blocks to create for intermmediate representation comparision
        DataParallel = True,         # Select between breaking single model onto
        Multi_GPU_replica = False,    # multiple GPUs or replicating model on 
                                      # multiple GPUs.Only select either of them
        Common_base_freeze = False,   # This freezes the common base to all the student models
        Collective_Base_gradient = False, # This passes gradients from all student back to the common base
        gpu=[0,1,2],              # For Resnet50(4 stu) recommended 2 GPUs, 
                                      # For Resnet101(4 stu) 2 GPUs, Resnet152(5 stu) 3 GPUs
        )

# Dataset Settings

'''
Choose dataset from "MNIST", "CIFAR10", "CIFAR100", "Fashion-MNIST"
                    "SVHN", "STL10", "Caltech", "Imagenet"
'''

dataset_cfg = dict(
    id_cfg=dict(
        root= "../data",
        name= "Imagenet",
        num_classes= 1000,
        download= False    # Keep true to download dataset through torch API
    ),
    train_cfg=dict(
        batch_size=256,
        shuffle=True,
        num_workers=20
    ),
    val_cfg=dict(
        batch_size=128,
        shuffle=False,
        num_workers=8
    )
)

# Model Training Settings

train_cfg = dict(
    optimizer=dict(
        name='Adam',
        lr=0.001,
        weight_decay=1e-4
    ),
    criterion=dict(
        L1='CrossEntropyLoss',    # Loss type for normal label loss 
        L2="KL_Loss",             # Loss for teacher, student probability comparision
        L3="MSELoss"              # Loss type for Intermmediate representation loss
    ),

    scheduler=dict(
        name='ReduceLROnPlateau',    # Select from LambdaLR, StepLR, MultiStepLR, 
                                     # ExponentialLR, ReduceLROnPlateau, CylicLR
       #patience=1,                   # For ReduceLROnPlateau
        #factor=0.1,
        #mode="max",
        #step_size=15,
        #exp_gamma=0.1,
        #verbose=True
        milestones=[50,70,90],   # For MultiStepLR
        last_epoch=-1,
        gamma=0.5
    ),


    teacher_pretraining= False,
    pretraining_epochs= 10,             # epochs for which to pretrain the pseudo teacher on
    KL_loss_temperature = 3,            # Temperature for creating softened log softmax for KL loss 
    test_interval = 10,
    plot_accuracy_graphs=True,
    epochs=100,
    training_store_root="../Model_storage"
)


# Training Resume settings
# Select from either resuming training or validating model on test set 

Single_model_mode = None               # Use for training baseline single student model. Select from None,0,1,2,3 ..

Train_resume = False
Validate_only = False
Validate_student_no = 0                 # This represents the version of student model you want to validate
Load_run_id = '01_19_19_10'
Load_Epoch = 2
