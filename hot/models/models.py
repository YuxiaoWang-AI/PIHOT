import torch
import torch.nn as nn
from . import resnet, resnext, mobilenet, hrnet
from hot.lib.nn import SynchronizedBatchNorm2d
import math
BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label > 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None, use_contrastive=False):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.use_contrastive = use_contrastive
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):

        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                # pass
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                img_feature = self.encoder(feed_dict['img_data'], return_feature_maps=True)
                inpaint_feature = self.encoder(feed_dict['inpaint_label'], return_feature_maps=True)
                pred, pred_p, pred_feat, pred_depth, pred_inpaint = self.decoder(img_feature, feed_dict['depth_label'], inpaint_feature)
            loss = self.crit(pred, feed_dict['seg_label']) # crit=NLLLoss
            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            img_feature = self.encoder(feed_dict['img_data'], return_feature_maps=True)
            inpaint_feature = self.encoder(feed_dict['inpaint_label'], return_feature_maps=True)
            pred, pred_b = self.decoder(img_feature, feed_dict['depth_label'], inpaint_feature, segSize=segSize)
            return pred, pred_b


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(cfg, arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                with_part=cfg.MODEL.with_part, # True
                use_softmax=use_softmax) # False
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


def conv3x3_bn(in_planes, out_planes, stride=1):
    "3x3 convolution + BN"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        BatchNorm2d(out_planes),
    )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


class AttentionIter(nn.Module):
    """docstring for AttentionIter"""
    def __init__(self, nChannels, LRNSize=1, IterSize=1):
        super(AttentionIter, self).__init__()
        self.nChannels = nChannels
        self.LRNSize = LRNSize
        self.IterSize = IterSize
        self.bn = nn.BatchNorm2d(self.nChannels)
        self.U = nn.Conv2d(self.nChannels, 1, 1, 1, 0)
        # self.spConv = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
        # self.spConvclone = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
        # self.spConvclone.load_state_dict(self.spConv.state_dict())
        _spConv_ = nn.Conv2d(1, 1, self.LRNSize, 1, 0)
        _spConv = []
        for i in range(self.IterSize):
            _temp_ = nn.Conv2d(1, 1, self.LRNSize, 1, 0)
            _temp_.load_state_dict(_spConv_.state_dict())
            _spConv.append(nn.BatchNorm2d(1))
            _spConv.append(_temp_)
        self.spConv = nn.ModuleList(_spConv)

    def forward(self, x):
        x = self.bn(x)
        u = self.U(x)
        out = u
        for i in range(self.IterSize):
            # if (i==1):
            # 	out = self.spConv(out)
            # else:
            # 	out = self.spConvclone(out)
            out = self.spConv[2*i](out)
            out = self.spConv[2*i+1](out)
            out = torch.sigmoid(u+out)
        return (x * out.expand_as(x)), out

class PartSegm(nn.Module):
    """docstring for AttentionIter"""
    def __init__(self, nChannels, num_class=1, use_conv=False, use_contrastive=False):
        super(PartSegm, self).__init__()
        self.nChannels = nChannels
        self.num_class = num_class
        self.use_conv = use_conv
        self.use_contrastive = use_contrastive
        self.sm_dim = 'channel'
        self.combine_res = False
        self.multihead_attn = nn.MultiheadAttention(nChannels, 8)

        self.depth_conv = nn.Sequential(
            conv3x3_bn_relu(1, nChannels // 4, 1),
            conv3x3_bn_relu(nChannels // 4, nChannels // 2, 1),
            conv3x3_bn_relu(nChannels // 2, nChannels, 1),
        )
        if self.combine_res: # false
            cbr = [conv3x3_bn_relu(nChannels, nChannels, 1) for _ in range(num_class)]
            self.cbr = nn.ModuleList(cbr)
            conv_last = [nn.Conv2d(nChannels, 1, 1, 1, 0) for _ in range(num_class)]
            self.conv_last = nn.ModuleList(conv_last)
        else:
            self.cbr = nn.Sequential(conv3x3_bn_relu(nChannels, nChannels, 1),
                                     conv3x3_bn_relu(nChannels, nChannels // 2, 1),
                                     conv3x3_bn_relu(nChannels // 2, nChannels // 4, 1),
                                     nn.Conv2d(nChannels // 4, num_class, 1, 1, 0))
            
        if use_conv:
            self.conv1x1 = nn.Conv2d(nChannels, 1, kernel_size=1)
        
        self.f_k_conv = conv3x3_bn_relu(nChannels, nChannels, 1)
        self.f_v_conv = conv3x3_bn_relu(nChannels, nChannels, 1)

    def forward(self, x, x_part, x_depth, x_inpaint_pre_conv):
        B_origin, C_origin, H_origin, W_origin = x.shape
        
        x_temp = x.detach()
        x_depth_temp = x_depth.unsqueeze(1).detach()
        
        x_temp = nn.functional.interpolate(x_temp, size=(15, 15), mode="bilinear")
        x_depth_temp = nn.functional.interpolate(x_depth_temp, size=(15, 15), mode="bilinear")
        x_temp_1 = self.f_k_conv(x_temp)
        x_temp_2 = self.f_v_conv(x_temp)
        
        x_depth_temp = self.depth_conv(x_depth_temp)
        B, C, H, W = x_temp.shape
        x_temp_1 = x_temp.reshape(B, C, -1).permute(2, 0, 1)
        x_temp_2 = x_temp.reshape(B, C, -1).permute(2, 0, 1)
        x_depth_temp = x_depth_temp.reshape(B, C, -1).permute(2, 0, 1)
        
        x_attention, _ = self.multihead_attn(x_depth_temp, x_temp_1, x_temp_2)
        
        x_attention = x_attention.permute(1, 2, 0).reshape(B, C, H, W)
        x_attention = nn.functional.interpolate(x_attention, size=(H_origin, W_origin), mode="bilinear")

        x = x * (x_depth.unsqueeze(1) / 3.0) + x
        x = x + x_attention * 0.1
        
        x = self.cbr(x)
        return x, x_part, None

# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, with_binary=False, with_part=False, use_contrastive=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax # false
        self.with_binary = with_binary # false
        self.with_part = with_part # ture
        self.use_contrastive = use_contrastive # false

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
       
        self.cbr_part = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        self.bbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        
        self.multihead_attn = nn.MultiheadAttention(fc_dim // 4, 8)

        if with_part:
            self.part_branch = PartSegm(fc_dim // 4, num_class, use_contrastive)
        
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

        # binary conv
        self.conv_binary = nn.Conv2d(fc_dim // 4, 2, 1, 1, 0)

    def forward(self, conv_out, x_depth, inpaint_out, segSize=None):
        conv5 = conv_out[-1]
        
        inpaint_feature = inpaint_out[-1]
        
        x_temp = self.cbr(conv5) 
        x = x_temp.detach()

        x_inpaint = self.bbr(inpaint_feature)
        
        B_origin, C_origin, H_origin, W_origin = x_inpaint.shape
        
        x_inpaint= nn.functional.interpolate(x_inpaint, size=(15, 15), mode="bilinear")
        x= nn.functional.interpolate(x, size=(15, 15), mode="bilinear")
        B, C, H, W = x_inpaint.shape
        x = x.reshape(B, C, -1).permute(2, 0, 1)
        x_inpaint = x_inpaint.reshape(B, C, -1).permute(2, 0, 1)
        x, _ = self.multihead_attn(x_inpaint, x, x+x_inpaint)
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        x = nn.functional.interpolate(x, size=(H_origin, W_origin), mode="bilinear")
        
        x = x_temp + x * 0.1
        if self.with_part: # True
            x_part = self.cbr_part(conv5)
            x, x_part, x_feat = self.part_branch(x, x_part, x_depth, None) # part attention operation
        else:
            x = self.conv_last(x)


        if self.with_binary: # false
            # binary contact prediction
            x_b = self.conv_binary(x)
            # semantic contact prediction
            x = self.conv_last(x)

        if self.use_softmax: # is True during inference # false
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            if self.with_part:
                x_part = nn.functional.interpolate(
                         x_part, size=segSize, mode='bilinear', align_corners=False)
                x_part = nn.functional.softmax(x_part, dim=1)
                return x, x_part
            else:
                return x, None
        else:
            x = nn.functional.log_softmax(x, dim=1)
            if self.with_part: # true
                x_part = nn.functional.log_softmax(x_part, dim=1)
                return x, x_part, x_feat, x_depth, x_inpaint
            else:
                return x, None, None


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x, None
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x, None, None


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x, None

        x = nn.functional.log_softmax(x, dim=1)

        return x, None, None
