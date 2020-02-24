# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
import torch.nn.functional as F
import numpy as np
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def KL_between_multivariate_gaussian(z_mu, z_var, c_mu, c_var):
    epsi = 1e-6

    det_z_var = torch.prod(z_var,1)
    det_c_var = torch.prod(c_var,1)
    inverse_c_var = 1 / (c_var+epsi)

    det_term = torch.unsqueeze(torch.log(det_c_var+epsi), 0) - torch.unsqueeze(torch.log(det_z_var+epsi), 1) #batchsize, num_class
    trace_term = torch.mm(z_var, inverse_c_var.t())#batchsize, num_class
    z_mu = torch.unsqueeze(z_mu, 1)
    c_mu = torch.unsqueeze(c_mu, 0)
    c_var = torch.unsqueeze(c_var, 0)
    diff = (z_mu-c_mu)**2
    m_dist_term = torch.sum(diff / (c_var+epsi), -1)#batchsize, num_class

    KL_divergence = 0.5*(det_term+trace_term+m_dist_term)
    return KL_divergence


class AttributeRecogModule(nn.Module):
    def __init__(self, in_planes, num_class):
        super(AttributeRecogModule, self).__init__()
        self.in_planes = in_planes
        self.attention_conv = nn.Conv2d(in_planes, 1, 1)
        weights_init_kaiming(self.attention_conv)
        self.classifier = nn.Linear(in_planes, num_class)
        self.classifier.apply(weights_init_classifier)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        a = self.attention_conv(x)
        a = torch.sigmoid(a)
        a = a.view(b, t, 1, x.size(2), x.size(3))

        atten = a.expand(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = atten * x.view(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat = self.gap(global_feat)
        # global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = global_feat.view(b, t, -1)
        global_feat = F.relu(torch.mean(global_feat, 1))
        y = self.classifier(global_feat)
        return y,a

class MultiAttributeRecogModule(nn.Module):
    def __init__(self, in_planes, num_classes=[]):
        super(MultiAttributeRecogModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes // 2
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 1)
        self.bn = nn.BatchNorm2d(self.out_planes)
        self.attr_recog_modules = nn.ModuleList([AttributeRecogModule(self.out_planes, n) for n in num_classes])
    def forward(self, x, b, t):
        ys = []
        attens = []
        local_feature = self.conv(x)
        local_feature = F.relu(self.bn(local_feature))
        local_feature = local_feature.view(b, t, self.out_planes, local_feature.size(2), local_feature.size(3))
        for m in self.attr_recog_modules:
            y, a = m(local_feature)
            ys.append(y)
            attens.append(a)
        return ys, torch.cat(attens, 2)

class MultiAttributeRecogModuleBCE(nn.Module):
    def __init__(self, in_planes, num_classes=[]):
        super(MultiAttributeRecogModuleBCE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 1)
        weights_init_kaiming(self.conv)
        self.bn = nn.BatchNorm2d(self.out_planes)
        weights_init_kaiming(self.bn)
        self.attention_conv = nn.Conv2d(self.out_planes, 1, 3, padding=1)
        weights_init_kaiming(self.attention_conv)
        self.classifier = nn.Linear(in_planes, sum(num_classes))
        self.classifier.apply(weights_init_classifier)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, b, t):
        local_feature = self.conv(x)
        local_feature = F.relu(self.bn(local_feature))
        a = self.attention_conv(local_feature)
        a = torch.sigmoid(a)
        a = a.view(b, t, 1, x.size(2), x.size(3))
        # a = torch.nn.functional.softmax(a, 1)

        atten = a.expand(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = atten * x.view(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat = self.gap(global_feat)
        # global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = global_feat.view(b, t, -1)
        global_feat = F.relu(torch.mean(global_feat, 1))
        if self.training:
            y = self.classifier(global_feat)
        else:
            y = None
        return y, atten

class VideoBaseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(VideoBaseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        # self.attention_conv = nn.Conv2d(self.in_planes, 1, 3, padding=1)
        # weights_init_kaiming(self.attention_conv)

        self.posterior_log_sigma1 = nn.Conv2d(self.in_planes, 512, kernel_size=1, stride=1, padding=0,
                                              bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.posterior_log_sigma2 = nn.Conv2d(self.in_planes, 512, kernel_size=1, stride=1, padding=0,
                                              bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.posterior_log_sigma = nn.Linear(512, self.in_planes, bias=True)

        # self.shortcut = nn.Parameter(torch.randn(1, self.in_planes, 1, 1).cuda())
        # nn.init.kaiming_normal_(self.shortcut, a=0, mode='fan_in'.
        self.shortcut = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.prior_mu = nn.Parameter(torch.randn(self.num_classes, self.in_planes).cuda())  #
        # nn.init.constant_(self.prior_mu, 0.0)
        nn.init.kaiming_normal_(self.prior_mu, a=0, mode='fan_in')
        self.prior_log_sigma = nn.Parameter(torch.ones(self.num_classes, self.in_planes).cuda()) * 0.54  #
        self.softplus = nn.Softplus()

        self.posterior_log_sigma1.apply(weights_init_classifier)
        self.posterior_log_sigma2.apply(weights_init_classifier)
        self.posterior_log_sigma.apply(weights_init_classifier)
        self.shortcut.apply(weights_init_classifier)

        self.bottleneck_mu = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_mu.bias.requires_grad_(False)  # no shift
        self.bottleneck_var = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_var.bias.requires_grad_(False)  # no shift

        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        # self.relu = nn.ReLU()

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        feature_map = self.base(x)
        global_feat = self.gap(feature_map)  # (b, 2048, 1, 1)
        posterior_mu = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feature_map_deatch = feature_map.detach()
        global_feat_detach = global_feat.detach()
        # temp_sigma = (feature_map_deatch - global_feat_detach)**2
        posterior_log_sigma1 = self.posterior_log_sigma1(feature_map)
        posterior_log_sigma1 = self.bn1(posterior_log_sigma1)
        posterior_log_sigma1 = self.tanh(posterior_log_sigma1)
        posterior_log_sigma2 = self.posterior_log_sigma2(feature_map)
        posterior_log_sigma2 = self.bn2(posterior_log_sigma2)
        posterior_log_sigma2 = self.tanh(posterior_log_sigma2)

        posterior_log_sigma = posterior_log_sigma1 * posterior_log_sigma2

        posterior_log_sigma = self.gap(posterior_log_sigma)
        posterior_log_sigma = posterior_log_sigma.view(global_feat.shape[0], -1)

        sigma_shortcut = self.shortcut(feature_map)
        sigma_shortcut = self.gap(sigma_shortcut)
        sigma_shortcut = sigma_shortcut.view(global_feat.shape[0], -1)

        posterior_log_sigma = self.posterior_log_sigma(posterior_log_sigma)
        posterior_log_sigma = posterior_log_sigma + sigma_shortcut

        # posterior_mu = self.bottleneck_mu(posterior_mu)
        # posterior_log_sigma_ce = self.bottleneck_var(posterior_log_sigma)
        posterior_sigma = self.softplus(posterior_log_sigma)

        # posterior_log_sigma_ce = self.bottleneck_var(posterior_log_sigma)
        # posterior_sigma_ce = self.relu(posterior_log_sigma_ce)

        prior_mu = self.prior_mu
        prior_sigma = self.softplus(self.prior_log_sigma)
        # prior_sigma = self.prior_log_sigma

        feat = torch.cat((posterior_mu, torch.sqrt(posterior_sigma+1e-6)), 1)
        feat_for_trip = torch.cat((posterior_mu, posterior_sigma), 1)

        if self.training:
            cls_score = -KL_between_multivariate_gaussian(posterior_mu, posterior_sigma, prior_mu, prior_sigma)
            return cls_score, feat, posterior_mu, posterior_sigma, prior_mu, prior_sigma  # global feature for triplet loss
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
