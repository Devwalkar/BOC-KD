import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
from torch.autograd import Variable 
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


def Growth_rate_computer(no_blocks = 3,
                           no_students = 4,
                           original_growth_rate =32
                           ):

    Growth_rate_list = []

    if no_blocks==1:
        for j in range(no_students):
            Growth_rate_list.append([original_growth_rate,
                                        original_growth_rate,
                                        int(original_growth_rate*(float(no_students-j)/no_students))
                                        ])
    
    elif no_blocks==2:
        for j in range(no_students):
            Growth_rate_list.append([original_growth_rate,
                                        int(original_growth_rate*(float(no_students-j)/no_students)),
                                        int(original_growth_rate*(float(no_students-j)/no_students))
                                        ])       

    elif no_blocks==3:
        for j in range(no_students):
            Growth_rate_list.append([int(original_growth_rate*(float(no_students-j)/no_students)),
                                        int(original_growth_rate*(float(no_students-j)/no_students)),
                                        int(original_growth_rate*(float(no_students-j)/no_students))
                                        ])    

    else:
        raise IndexError("No blocks can only be 1,2 or 3")

    return Growth_rate_list


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class BaseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False,
                 parallel=False,gpus=[0,1]):

        super(BaseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_layers = int(block_config[0])
        block = _DenseBlock(
                    num_layers= num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient
                )

        self.features.add_module('denseblock1', block)
        num_features = num_features + num_layers * growth_rate

        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module('transition1', trans)
        num_features = num_features // 2

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.num_features = num_features

    def forward(self, x):
        return self.features(x)


class DenseNet_Student(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, growth_rates=[32,32,32], block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, 
                 memory_efficient=False, parallel=False, gpus=[0,1]):

        super(DenseNet_Student, self).__init__()

        self.blocks = []
        self.transition_layers = []

        # Each denseblock
        num_features = num_init_features
        for i, (num_layers, growth_rate) in enumerate(zip(block_config[1:],growth_rates)):
            '''
            if parallel:
                self.blocks.append(torch.nn.DataParallel(_DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient
                ).to(device),device_ids=gpus))
            else:
            '''
            self.blocks.append(_DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            ).to(device))               

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                '''
                if parallel:
                    self.transition_layers.append(torch.nn.DataParallel(_Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2).to(device),device_ids=gpus))
                else:
                '''
                self.transition_layers.append(_Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2).to(device))

                num_features = num_features // 2

        # Final batch norm
        self.final_norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x_ = self.blocks[0](x)
        x1 = self.transition_layers[0](x_)
        x_ = self.blocks[1](x1)
        x2 = self.transition_layers[1](x_)
        x_ = self.blocks[2](x2)
        x3 = self.transition_layers[2](x_)
        x3 = self.final_norm(x3)
        out = F.relu(x3, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out,[x1,x2,x3]


class BIO_DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes = 1000, 
                 memory_efficient=False, no_blocks = 3, no_students = 4, 
                 parallel=False,gpus=[0,1], Base_freeze = False
                 ):

        super(BIO_DenseNet, self).__init__()

        # Initializing the common base resent model

        self.BaseNet = BaseNet(growth_rate=growth_rate,
                               block_config=block_config,
                               num_init_features=num_init_features,
                               bn_size=bn_size,
                               drop_rate=drop_rate,
                               memory_efficient=memory_efficient,
                               parallel=parallel,
                               gpus=gpus)

        self.BaseNet = self.BaseNet.to(device) if not parallel else torch.nn.DataParallel(self.BaseNet.to(device),
                                                                                          device_ids =gpus)
        if Base_freeze:
            self.base_freezer()
                                                                                         
        growth_rate_list = Growth_rate_computer(no_blocks=no_blocks,
                                                no_students=no_students,
                                                original_growth_rate=growth_rate
                                                )

        self.pretrain_mode = False
        self.no_students = no_students
        self.no_blocks = no_blocks
        self.student_num_features = self.BaseNet.module.num_features if parallel else self.BaseNet.num_features

        # Initializing student models

        self.student_models = []

        for growth_rates in growth_rate_list:

            Student_M = DenseNet_Student(growth_rates=growth_rates,
                                         block_config=block_config,
                                         num_init_features=self.student_num_features,
                                         bn_size=bn_size,
                                         drop_rate=drop_rate,
                                         num_classes=num_classes,
                                         memory_efficient=memory_efficient,
                                         parallel=parallel,
                                         gpus=gpus)

            if not parallel:
                self.student_models.append(Student_M.to(device))
            else:
                self.student_models.append(torch.nn.DataParallel(Student_M.to(device),device_ids=gpus))
        
        # Initializing contribution weights for teacher output

        self.weights = Variable(torch.Tensor((float(1)/self.no_students)*np.ones([1,self.no_students,1])),
                                requires_grad= True).to(device)

    def base_freezer(self):

        print("----------- Freezing common base model")

        for m in self.BaseNet.parameters():
            if m.requires_grad:
                m.requires_grad = False

    def pretrain(self):
        self.pretrain_mode = True

    def student_version(self):
        self.pretrain_mode = False

    def _forward_teacher_pretrain(self, x):

        x = self.BaseNet(x)

        Teacher_out,_ = self.student_models[0](x)         

        return Teacher_out

    def _forward_student(self, x):

        x = self.BaseNet(x)

        x_copy = Variable(x.clone(),requires_grad=False)

        Student_final_outs = []
        Student_intermmediate_reps = [[] for _ in range(self.no_blocks)]

        for i in range(self.no_students):
            if i==0:
                Final_out,Inter_reps = self.student_models[0](x)
                Combined_student_outs = Final_out.unsqueeze(1)
            else:
                Final_out,Inter_reps = self.student_models[i](x_copy)
                Combined_student_outs = torch.cat((Combined_student_outs,Final_out.unsqueeze(1)),dim=1) # Combined_student_outs shape : (Batch_size,no_students,num_classes) 

            Student_final_outs.append(Final_out)
            
            for j in range(self.no_blocks):
                Student_intermmediate_reps[j].append(Inter_reps[j])

        Teacher_out = torch.sum((self.weights*Combined_student_outs),dim=1).squeeze(1)               # Teacher out shape : (Batch_size, num_classes)

        Student_final_outs = [Teacher_out] + Student_final_outs           

        return Student_final_outs,Student_intermmediate_reps

    def forward(self, x):
        if self.pretrain_mode:
            return self._forward_teacher_pretrain(x)
        else:
            return self._forward_student(x)

def count_parameters(model):

    total = 0
    trainable = 0
    for param in model.parameters():
        temp = param.numel()
        total += temp
        if param.requires_grad:
            trainable += temp

    return (total,trainable)

def parameter_model_counter(base_model,student_models):

    # Function to count individual student model parameters

    Base_model_count = count_parameters(base_model)
    Student_model_count = [count_parameters(model) for model in student_models]

    Original_model_params = Base_model_count[0]+Student_model_count[0][0]

    Total_params,Total_trainable_params = Base_model_count[0],Base_model_count[1]
    print("\nParameter count -->")
    for i,stu_count in enumerate(Student_model_count):
        print("Student {}: Total {} | Trainable: {} Compress_ratio: {:.4f}".format(i+1,Base_model_count[0]+stu_count[0],
                                                                               Base_model_count[1]+stu_count[1],
                                                            float(Base_model_count[0]+stu_count[0])/Original_model_params))
        Total_params+=stu_count[0]
        Total_trainable_params+=stu_count[1]
    
    print("Overall Teacher: Total: {} | Trainable: {}\n".format(Total_params,Total_trainable_params))


def pretrained_weight_formatter(Arch,parallel):

    base_weights = dict()

    if Arch == "Densenet121":
        Overall_model_dict = torch.load("../models/pretrained_weights/Densenet/densenet121.pth")

    elif Arch == "Densenet161":
        Overall_model_dict = torch.load("../models/pretrained_weights/Densenet/densenet161.pth")

    elif Arch == "Densenet169":
        Overall_model_dict = torch.load("../models/pretrained_weights/Densenet/densenet169.pth")

    elif Arch == "Densenet201":
        Overall_model_dict = torch.load("../models/pretrained_weights/Densenet/densenet201.pth")

    for key in Overall_model_dict.keys():

        K = True
        for f in ["denseblock1","transition1","conv0","pool0","relu0","norm0"]:
            if f in key:
                K = False
        
        if K:
            if parallel:
                base_weights["module."+key] = Overall_model_dict[key]
            else:
                base_weights[key] = Overall_model_dict[key]

    return base_weights


def _bio_densenet(arch, growth_rate, block_config, num_init_features,pretrained,
                  bn_size,num_classes,no_blocks,no_students,
                  parallel,gpus,Base_freeze,**kwargs
                  ):

    model = BIO_DenseNet(growth_rate=growth_rate, 
                         block_config=block_config,
                         num_init_features=num_init_features,
                         bn_size=bn_size, drop_rate=0, num_classes = num_classes, 
                         memory_efficient=False, no_blocks = no_blocks, no_students = no_students,
                         parallel=parallel,gpus=gpus, Base_freeze = Base_freeze,**kwargs)

    parameter_model_counter(model.BaseNet,model.student_models)

    if pretrained:
        state_dict = pretrained_weight_formatter(arch,parallel)
        model.BaseNet.load_state_dict(state_dict)
        print("----------- ImageNet pretrained weights successfully loaded")

    return model



def BIO_Densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _bio_densenet('Densenet121', 32, (6, 12, 24, 16), 64, pretrained,
                     **kwargs)


def BIO_Densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _bio_densenet('Densenet161', 48, (6, 12, 36, 24), 96, pretrained,
                     **kwargs)


def BIO_Densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _bio_densenet('Densenet169', 32, (6, 12, 32, 32), 64, pretrained,
                     **kwargs)


def BIO_Densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _bio_densenet('Densenet201', 32, (6, 12, 48, 32), 64, pretrained,
                     **kwargs)
