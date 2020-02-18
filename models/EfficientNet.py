import math
import torch
from torch import nn
from torch.autograd import Variable 
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

def width_multiplier_computer(no_students = 4,
                              original_width_multiplier = 1
                           ):

    Width_multiplier_list = []

    for j in range(no_students):
            Width_multiplier_list.append(float(original_width_multiplier*(float(no_students-j)/no_students)))    

    return Width_multiplier_list

class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class BaseNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(BaseNet, self).__init__()

        out_channels = _round_filters(32, width_mult)
        self.features = ConvBNReLU(3, out_channels, 3, stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.features(x)
        return x

class EfficientNet_Student(nn.Module):

    def __init__(self,out_width_mult=1.0, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet_Student, self).__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, out_width_mult)
        features_stage_1 = []
        features_stage_2 = []
        features_stage_3 = []

        in_channels = out_channels
        for j,(t, c, n, s, k) in enumerate(settings):
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)

            if j<=2:
                for i in range(repeats):
                    stride = s if i == 0 else 1
                    features_stage_1 += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                    in_channels = out_channels
            elif (2<j) and (j<=4):
                for i in range(repeats):
                    stride = s if i == 0 else 1
                    features_stage_2 += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                    in_channels = out_channels
            else:
                for i in range(repeats):
                    stride = s if i == 0 else 1
                    features_stage_3 += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                    in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features_stage_3 += [ConvBNReLU(in_channels, last_channels, 1)]

        self.features_stage_1 = nn.Sequential(*features_stage_1)
        self.features_stage_2 = nn.Sequential(*features_stage_2)
        self.features_stage_3 = nn.Sequential(*features_stage_3)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.features_stage_1(x)
        x2 = self.features_stage_2(x1)
        x3 = self.features_stage_3(x2)
        x_out = x3.mean([2, 3])
        x_out = self.classifier(x_out)
        return x_out,[x1,x2,x3]


class BIO_EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000,
                 Base_freeze= False,
                 no_students = 4,
                 no_blocks = 3,             # Select only from 3,2,1
                 parallel = False,
                 gpus = [0,1],
                 Common_Base = False,
                 Single_model = 0                 
                 ):
        super(BIO_EfficientNet, self).__init__()

        width_multipliers = width_multiplier_computer(no_students=no_students,
                                                      original_width_multiplier=width_mult)

        self.no_students = no_students
        self.no_blocks = no_blocks
        self.pretrain_mode = False
        self.single_model_mode = False
        self.Common_Base = Common_Base
        self.single_model = Single_model

        if Common_Base:
            print("---------- Passing collective gradients through Common Base")

        self.BaseNet = BaseNet(width_mult=width_mult,
                               depth_mult=depth_mult,
                               dropout_rate=dropout_rate,
                               num_classes=num_classes
                               )
        self.BaseNet = self.BaseNet.to(device) if not parallel else torch.nn.DataParallel(self.BaseNet.to(device),
                                                                                               device_ids =gpus)
        if Base_freeze:
            self.base_freezer()

        # Initializing student models

        self.student_models = []

        for width_multiplier in width_multipliers:
            Student_M = EfficientNet_Student(out_width_mult=width_mult,
                                             width_mult=width_multiplier,
                                             depth_mult=depth_mult,
                                             dropout_rate=dropout_rate,
                                             num_classes=num_classes
                                             )
        
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

    def Single_mode(self):
        self.single_model_mode = True

    def student_version(self):
        self.pretrain_mode = False

    def _forward_teacher_pretrain(self, x):

        x = self.BaseNet(x)

        Teacher_out,_ = self.student_models[0](x)         

        return Teacher_out

    def _forward_student_model(self, x):

        x = self.BaseNet(x)

        Teacher_out,_ = self.student_models[int(self.single_model)](x)         

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
                Inter_reps = [single_rep.detach() for single_rep in Inter_reps]
            else:
                Final_out,Inter_reps = self.student_models[i](x_copy) if self.Common_Base else self.student_models[i](x)
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
        elif self.single_model_mode:
            return self._forward_student_model(x)
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

    if Arch == "efficientnet_b0":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b0.pth")

    elif Arch == "efficientnet_b1":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b1.pth")

    elif Arch == "efficientnet_b2":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b2.pth")

    elif Arch == "efficientnet_b3":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b3.pth")

    elif Arch == "efficientnet_b4":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b4.pth")

    elif Arch == "efficientnet_b5":
        Overall_model_dict = torch.load("../models/pretrained_weights/Efficientnet/efficientnet-b5.pth")

    else:
        raise ImportError("Sorry! Pretrained weights avilable only from B0 to B5")

    for key in Overall_model_dict.keys():

        if "features.0" in key:
            New_key = key.replace("features.0","features")
        
            base_weights[New_key] = Overall_model_dict[key]

    return base_weights

def _bio_efficientnet(arch, pretrained,**kwargs):

    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = BIO_EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)

    parameter_model_counter(model.BaseNet,model.student_models) 

    if pretrained:
        state_dict = pretrained_weight_formatter(arch,parallel)
        model.BaseNet.load_state_dict(state_dict)
        print("----------- ImageNet pretrained weights successfully loaded")

    return model


def BIO_Efficientnet_B0(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b0', pretrained, **kwargs)


def BIO_Efficientnet_B1(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b1', pretrained, **kwargs)


def BIO_Efficientnet_B2(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b2', pretrained, **kwargs)


def BIO_Efficientnet_B3(pretrained=False,**kwargs):
    return _bio_efficientnet('efficientnet_b3', pretrained, **kwargs)


def BIO_Efficientnet_B4(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b4', pretrained, **kwargs)


def BIO_Efficientnet_B5(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b5', pretrained, **kwargs)


def BIO_Efficientnet_B6(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b6', pretrained, **kwargs)


def BIO_Efficientnet_B7(pretrained=False, **kwargs):
    return _bio_efficientnet('efficientnet_b7', pretrained, **kwargs)
