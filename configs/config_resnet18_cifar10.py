""" Template Configuration file for CIFAR10 training on Resnet18
"""

# DL model Architecture Settings

'''
Choose DL model from  "Resnet20", "Resnet34", "Resnet50", "Resnet101", "Resnet152", "Resnet32", "Resnet110"

'''
model = dict(
        name ="Resnet18",
        pretrained = False,            # Select between True and False
        No_students = 5,              # Number of student models to create for training
        No_blocks = 3,                # Number of blocks to create for intermmediate representation comparision
        DataParallel = True,          # Select between breaking single model onto
        Multi_GPU_replica = False,    # multiple GPUs or replicating model on 
                                      # multiple GPUs.Only select either of them
        Common_base_freeze = False,   # This freezes the common base to all the student models
        gpu=[0,1],                    # For Resnet50(4 stu) recommended 2 GPUs, 
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
        name= "CIFAR10",
        num_classes= 10,
        download= False    # Keep true to download dataset through torch API
    ),
    train_cfg=dict(
        batch_size=32,
        shuffle=True,
        num_workers=20
    ),
    val_cfg=dict(
        batch_size=32,
        shuffle=False,
        num_workers=8
    )
)

# Model Training Settings

train_cfg = dict(
    optimizer=dict(
        name='SGD',
        lr=0.01,
        weight_decay=1e-5,
        momentum=0.9
    ),
    criterion=dict(
        L1='CrossEntropyLoss',    # Loss type for normal label loss 
        L2="KL_Loss",             # Loss for teacher, student probability comparision
        L3="MSELoss"              # Loss type for Intermmediate representation loss
    ),

    Loss_contribution=dict(
        alpha = 0.4,              # Contribution ratio for Normal label loss
        beta = 0.3,               # Contribution ratio for Intermmediate loss
        gamma = 0.3               # Contribution ratio for KL Loss 
    ),

    scheduler=dict(
        name='ReduceLROnPlateau',    # Select from LambdaLR, StepLR, MultiStepLR, 
                                     # ExponentialLR, ReduceLROnPlateau, CylicLR
        patience=5,
        #step_size=15,
        #exp_gamma=0.1,
        verbose=True
    ),

    KL_loss_temperature = 3,            # Temperature for creating softened log softmax for KL loss 
    test_interval = 1,
    plot_accuracy_graphs=True,
    epochs=300,
    training_store_root="../Model_storage"
)


# Training Resume settings
# Select from either resuming training or validating model on test set 

Train_resume = False
Validate_only = False
Load_run_id = '01_03_23_49'
Load_Epoch = 3