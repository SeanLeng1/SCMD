# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet152_Weights, ViT_B_16_Weights
#from torchvision.models import resnet50, resnet18

from domainbed.lib import wide_resnet
import copy
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

#import clip
from clip_model import *
from matplotlib import pyplot as plt
import seaborn as sns

#swad
from domainbed.lib import swa_utils

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

### not done, maybe adding VGG module here
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
   
        checkpoint_path = hparams.get('checkpoint_path', None)
        if checkpoint_path is None:
            weights_50 = ResNet50_Weights.DEFAULT
            weights_18 = ResNet18_Weights.DEFAULT
            weights_152 = ResNet152_Weights.DEFAULT
        else:
            weights_50 = None
            weights_18 = None
            weights_152 = None

        rn152 = hparams.get('resnet152', False)
        if hparams['resnet18']:
            print("using resnet 18")
            self.network = torchvision.models.resnet18(weights = weights_18)
            self.n_outputs = 512
        elif rn152:
            print("using resnet 152")
            self.network = torchvision.models.resnet152(weights = weights_152)
            self.n_outputs = 2048
        else:
            print("using resnet 50")
            self.network = torchvision.models.resnet50(weights = weights_50)
            self.n_outputs = 2048       
        if checkpoint_path is not None:
            print("loading checkpoint from", checkpoint_path)
            self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()
        self.freeze_bn()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

        #architecture
        self.conv1 = self.network.conv1
        self.bn1 = self.network.bn1
        self.relu = self.network.relu
        self.maxpool = self.network.maxpool
        self.layer1 = self.network.layer1
        self.layer2 = self.network.layer2 
        self.layer3 = self.network.layer3
        self.layer4 = self.network.layer4
        self.avgpool = self.network.avgpool
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, projection = False):
        """Encode x into a feature vector of size n_outputs."""
        #return self.dropout(self.network(x))

        if not projection:
            return self.dropout(self.network(x))
          
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)

            feat1 = x # N 256 56 56
            x = self.layer2(x)

            feat2 = x # N 512 28 28
            x = self.layer3(x)

            feat3 = x # N 1024 14 14
            x = self.layer4(x) 

            feat4 = x # N 2048 7 7

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.dropout(x), [feat1, feat2, feat3, feat4]
   

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)

        # if not self.unfreeze_resnet_bn:
        #     self.freeze_bn()

        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class VisionTransformer(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        print("using vision transformer")
        super(VisionTransformer, self).__init__()
        self.network = torchvision.models.vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        del self.network.heads.head
        self.network.heads.head = Identity()
        self.n_outputs = 768
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x, projection = False):
        if not projection:
            return self.dropout(self.network(x))
        else:
            return self.dropout(self.network(x)), None


# AttentionPool2d from CLIP https://github.com/openai/CLIP/blob/main/clip/model.py
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)



class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # if hparams['backbone'] == 'clip':
    #     return CLIP_Featurizer(hparams)

    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError

# add a featurizer function for the student model
def Student_Featurizer(input_shape, hparams):
    print("constructing student model")
    if hparams.get('ViT', False):
            return VisionTransformer(input_shape, hparams)
            
    if hparams['student_model'] == 'resnet':
        return ResNet(input_shape, hparams)

    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError

class CLIP_Featurizer(nn.Module):
    def __init__(self, hparams):
        super(CLIP_Featurizer, self).__init__()
        #print(hparams['clip_backbone'])
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if hparams['clip_backbone'] not in clip.available_models():
            raise ValueError(f"backbone {hparams['clip_backbone']} not available")


        download_root = hparams.get('download_root', None)
        print(f'Using {hparams["clip_backbone"]}...')

        if download_root is None:
            self.clip_model = clip.load(hparams['clip_backbone'])[0].float()
        else:
            print(f'Loading from {download_root}...')
            self.clip_model = clip.load(download_root)[0].float()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # embedding dimensions based on CLIP paper https://arxiv.org/pdf/2103.00020.pdf
        if hparams['clip_backbone'] == 'RN50':
            self.n_outputs = 1024
        elif hparams['clip_backbone'] == 'RN101' or hparams['clip_backbone'] == 'ViT-B/32' or hparams['clip_backbone'] == 'ViT-B/16':
            self.n_outputs = 512
        elif hparams['clip_backbone'] == 'RN50x16' or hparams['clip_backbone'] == 'ViT-L/14' or hparams['clip_backbone'] == 'ViT-L/14@336px':
            self.n_outputs = 748
        elif hparams['clip_backbone'] == 'RN50x64':
            self.n_outputs = 1024
        elif hparams['clip_backbone'] == 'RN50x4':
            self.n_outputs = 640
        # attention dimension
        if hparams['clip_backbone'] == 'RN50' or hparams['clip_backbone'] == 'RN101':
            self.width = 2048
        elif hparams['clip_backbone'] == 'ViT-B/32' or hparams['clip_backbone'] == 'ViT-B/16':
            self.width = 768
        elif hparams['clip_backbone'] == 'ViT-L/14' or hparams['clip_backbone'] == 'ViT-L/14@336px':
            self.width = 1024
        elif hparams['clip_backbone'] == 'RN50x16':
            self.width = 3072
        elif hparams['clip_backbone'] == 'RN50x4':
            self.width = 3072
        elif hparams['clip_backbone'] == 'RN50x64':
            self.width = 4096
        # resnet does not need to return cls token
        if hparams['clip_backbone'] == 'RN50' or hparams['clip_backbone'] == 'RN101' or hparams['clip_backbone'] == 'RN50x16' or hparams['clip_backbone'] == 'RN50x64' \
            or hparams['clip_backbone'] == 'RN50x4':
            self.return_cls = False
        else:
            self.return_cls = True

    def forward(self, x):
        image_features, _, _, _, _= self.clip_model.encode_image(x)
        return image_features


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


# Borrowed from https://github.com/openai/CLIP
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)



def get_correlation_matrix(feature):
    feature_transpose = feature.transpose(1, 2)
    D_sqrt = torch.sqrt(feature.shape[1])
    correlation_matrix = torch.matmul(feature, feature_transpose) / D_sqrt
    return correlation_matrix

def dkd_loss(student_logits, teacher_logits, target, alpha = 1.0, beta = 8.0, temperature = 4.0, reduction = 'batchmean'):
    gt_mask = get_gt_mask(student_logits, target)
    other_mask = get_other_mask(student_logits, target)
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction = reduction) * (temperature ** 2) / target.shape[0]
    pred_teacher_part2 = F.softmax(teacher_logits / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(student_logits / temperature - 1000.0 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction = reduction) * (temperature ** 2) / target.shape[0]
    return alpha * tckd_loss + beta * nckd_loss

def get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim = 1, keepdim = True)
    t2 = (t * mask2).sum(dim = 1, keepdim = True)
    rt = torch.cat([t1, t2], dim = 1)
    return rt


def gram_matrix_loss(student_features, teacher_features, reduction='mean'):
    student_gram = torch.matmul(student_features, student_features.t())
    teacher_gram = torch.matmul(teacher_features, teacher_features.t())
    loss = F.mse_loss(student_gram, teacher_gram, reduction=reduction)
    return loss



# dynamic mask ratio
def calculate_mask_ratio(feature_loss, min_ratio, max_ratio):
    min_loss, max_loss = torch.min(feature_loss), torch.max(feature_loss)
    mask_ratios = min_ratio + (max_ratio - min_ratio) * (feature_loss - min_loss) / (max_loss - min_loss + 1e-7) # avoid division by zero
    return mask_ratios

# random mask
def create_random_mask(shape, mask_ratio):
    total_elements = 1
    for dimension in shape[1:]:
        total_elements *= dimension
    temp_mask = []
    for ratio in mask_ratio:
        false_count = int(total_elements * ratio)
        true_count = total_elements - false_count
        mask = [True] * true_count + [False] * false_count
        np.random.shuffle(mask)
        temp_mask.append(mask)
    bool_mask = torch.Tensor(temp_mask).bool()
    return bool_mask


# function to renormalize the image for CLIP
def denormalize(images, type="imagenet"):
    # sr_images [B, 3, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1).type_as(images)
    return std * images + mean

def normalize(images, type="clip"):
    # images [B, 3, h, w]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).type_as(images)
    return (images - mean) / std




class FitNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(FitNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transform = nn.Conv2d(in_features, out_features, 1, bias = False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student_feature, teacher_feature):
        if student_feature.dim() == 2:
            student_feature = student_feature.unsqueeze(2).unsqueeze(3)
            teacher_feature = teacher_feature.unsqueeze(2).unsqueeze(3)

        return (self.transform(student_feature) - teacher_feature).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, student_feature, teacher_feature):
        s_attn = F.normalize(student_feature.pow(2).mean(1).view(student_feature.size(0), -1))
        with torch.no_grad():
            t_attn = F.normalize(teacher_feature.pow(2).mean(1).view(teacher_feature.size(0), -1))
        return (s_attn - t_attn).pow(2).mean()
