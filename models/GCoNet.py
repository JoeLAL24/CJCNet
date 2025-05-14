import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
import fvcore.nn.weight_init as weight_init

# EnLayer：普通卷积，通道数变为64
class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x

# LatLayer：普通卷积，通道数变为64
class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x

# DSLayer：输出单通道的图像
class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


# half_DSLayer：输出图像通道为原来的1/4
class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(int(in_channel/2), int(in_channel/4), kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # kernel_size 通常为 7，确保感受野覆盖较大的空间区域
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: x (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化，(B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化，(B, 1, H, W)
        
        # 拼接池化特征
        x = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # 通过卷积和 Sigmoid 激活
        x = self.conv(x)  # (B, 1, H, W)
        return self.sigmoid(x)  # (B, 1, H, W)

# AllAttLayer：GAM模块
class AllAttLayer(nn.Module):
    def __init__(self, input_channels=512):

        super(AllAttLayer, self).__init__()
        # query_transform和key_transform：用于生成Query和Key向量
        self.query_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.key_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (input_channels ** 0.5)

        self.conv6 = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

        # 空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size=7)


        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        # 加入空间注意力模块
        x5 = self.spatial_attention(x5) * x5

        # x_query: B,C,HW
        # query_transfor本质上就是1*1卷积
        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: BHW, C
        # torch.transpose(x_query, 1, 2)：交换x的维度1和维度2,从B,C,HW到B,HW,C
        # .view(-1, C)表示为总共为两维，第二维是C，所以变成了BHW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C

        # x_key: B,C,HW
        # key_transform本质上也是1*1卷积
        x_key = self.key_transform(x5).view(B, C, -1)

        # x_key: C, BHW
        # 同x_key，只是最终变换形式不一样，变成了C，BHW
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # 注意力计算
        # W = Q^T K: B,HW,HW
        # x_w: BHW, BHW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW

        # x_w: BHW,B,HW
        x_w = x_w.view(B*H5*W5, B, H5*W5)

        # x_w: BHW, B
        # 提取最后一个维度的最大值，从BHW,B,HW变为BHW,B
        x_w = torch.max(x_w, -1).values # BHW, B

        # x_w: BHW, 1
        # 将最后一维平均化
        x_w = x_w.mean(-1)

        #x_w：B, HW
        x_w = x_w.view(B, -1) * self.scale # B, HW

        x_w = F.softmax(x_w, dim=-1) # B, HW

        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        # x5: B,C,H,W
        return x5

class CoAttLayer(nn.Module):
    def __init__(self, input_channels=512):

        super(CoAttLayer, self).__init__()

        self.all_attention = AllAttLayer(input_channels)
        self.conv_output = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.fc_transform = nn.Linear(input_channels, input_channels)
        self.conv1x1 = Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv3x3 = Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv5x5 = Conv2d(input_channels, input_channels, kernel_size=5, padding=2)

        for layer in [self.conv_output, self.conv_transform, self.fc_transform]:
            weight_init.c2_msra_fill(layer)
    
    def forward(self, x5):

        # x5: B,C,H,W
        if self.training:
            f_begin = 0
            # f_end：批数的一半
            f_end = int(x5.shape[0] / 2)

            s_begin = f_end
            #s_end：批数
            s_end = int(x5.shape[0])

            # x5_1：前面一半批数
            x5_1 = x5[f_begin: f_end]
            # x5_2：后面一半批数
            x5_2 = x5[s_begin: s_end]

            # x5_new_1：经过AllAttLayer层的特征，2同理
            # x5_new_1：B/2,C,H,W
            x5_new_1 = self.all_attention(x5_1)
            x5_new_2 = self.all_attention(x5_2)

            # x5_1_proto：1，C
            x5_1_proto = torch.mean(x5_new_1, (0, 2, 3), True).view(1, -1)
            # x5_1_proto：1，C，1，1
            x5_1_proto = x5_1_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            # x5_2_proto：1，C
            x5_2_proto = torch.mean(x5_new_2, (0, 2, 3), True).view(1, -1)
            # x5_2_proto：1，C，1，1
            x5_2_proto = x5_2_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            # x5_11：F11
            x5_11 = x5_1 * x5_1_proto
            # x5_22：F22
            x5_22 = x5_2 * x5_2_proto
            # weighted_x5：cat(F11, F22)
            weighted_x5 = torch.cat([x5_11, x5_22], dim=0)

            # x5_12：F12
            x5_12 = x5_1 * x5_2_proto
            # x5_21：F21
            x5_21 = x5_2 * x5_1_proto
            # neg_x5：cat(F12, F21)
            neg_x5 = torch.cat([x5_12, x5_21], dim=0)
        else:

            x5_new = self.all_attention(x5)
            x5_proto = torch.mean(x5_new, (0, 2, 3), True).view(1, -1)
            x5_proto = x5_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            weighted_x5 = x5 * x5_proto #* cweight
            neg_x5 = None
        return weighted_x5, neg_x5

class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GCM(nn.Module):  # 输入通道首先经过四个卷积层的特征提取，并采用torch.cat()进行连接，最后和输入通道的残差进行相加
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.conv_cat = BasicConv2d(6*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4, x5), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class SpatialAtt(nn.Module):
    # 输入特征，返回经过处理后的该层特征
    def __init__(self, in_channels):
        super(SpatialAtt, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            nn.BatchNorm2d(64, momentum=0.001),
            nn.ReLU(inplace=True),
        )
        self.SpatialAtt = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, the_enc, last_dec):
        [_, _, H, W] = the_enc.size()
        last_dec_1 = F.interpolate(last_dec, size=(H, W), mode='bilinear', align_corners=False)
        spatial_att = self.SpatialAtt(last_dec_1)
        the_enc = self.enc_fea_proc(the_enc)
        the_enc1 = the_enc*spatial_att
        return the_enc1

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class GINet(nn.Module):
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(GINet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode

        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

        self.pred_layer = half_DSLayer(512)

        self.co_x5 = CoAttLayer()

        self.gcm1 = GCM(64, 64)

        self.spatial1 = SpatialAtt(64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_block = SEBlock(512)
        self.classifier = nn.Linear(512, 291)

        for layer in [self.classifier]:
            weight_init.c2_msra_fill(layer)

    def set_mode(self, mode):
        self.mode = mode

    # 将x上采样到y的尺寸，再与y相加
    # x：高层低分辨率，y：低层高分辨率
    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    # 将pred上采样到feat的尺寸
    def _fg_att(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = F.interpolate(pred,
                             size=(H, W),
                             mode='bilinear',
                             align_corners=True)
        return feat * pred

    def forward(self, x):
        if self.mode == 'train':
            preds = self._forward(x)
        else:
            with torch.no_grad():
                preds = self._forward(x)

        return preds

    def _forward(self, x):
        [_, _, H, W] = x.size()
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)

        x5 = self.se_block(x5)
        _x5 = self.avgpool(x5)
        _x5 = _x5.view(_x5.size(0), -1)
        pred_cls = self.classifier(_x5)

        # weighted_x5：cat(F11, F22)
        # neg_x5：cat(F12, F21)
        weighted_x5, neg_x5 = self.co_x5(x5)
       
        cam = torch.mean(weighted_x5, dim=1).unsqueeze(1)
        cam = cam.sigmoid()
        if self.training:
            ########## contrastive branch #########
            cat_x5 = torch.cat([weighted_x5, neg_x5], dim=0)
            pred_x5 = self.pred_layer(cat_x5)
            pred_x5 = F.interpolate(pred_x5,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)

        ########## Up-Sample ##########
        preds = []
        p5 = self.toplayer(weighted_x5)
        _pred = cam
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p4 = self._upsample_add(p5, self.latlayer4(x4)) 
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p3 = self._upsample_add(p4, self.latlayer3(x3)) 
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p2 = self._upsample_add(p3, self.latlayer2(x2)) 
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        x1_spa = self.spatial1(x1, p2)
        x1_gcm = self.gcm1(x1_spa)
        p1 = self._upsample_add(p2, self.latlayer1(x1_gcm)) 
        p1 = self.enlayer1(p1)
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        if self.training:
            return preds, pred_cls, pred_x5
        else:
            return preds


class GCoNet(nn.Module):
    def __init__(self, mode='train'):
        super(GCoNet, self).__init__()
        self.co_classifier = vgg16(pretrained=True).eval()
        self.ginet = GINet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.ginet.set_mode(self.mode)

    def forward(self, x):
        ########## Co-SOD ############
        preds = self.ginet(x)

        return preds

