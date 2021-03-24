from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.layers import Conv2d, ConvTranspose2d, GroupNorm, DFConv2d_guide

@registry.ROI_KEYPOINT_PREDICTOR.register("KeypointRCNNPredictor")
class KeypointRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNPredictor, self).__init__()
        self.in_channels = in_channels
        self.num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        self.num_convs = 4
        self.point_feat_channels = 32
        self.conv_out_channels = self.point_feat_channels * self.num_keypoints
        conv_kernel_size = 3
        conv_kernel_size1 = 5
        deconv_kernel_size = 4
        # deconv_kernel = 4
        # self.kps_score_lowres = layers.ConvTranspose2d(
        #     input_features,
        #     num_keypoints,
        #     deconv_kernel,
        #     stride=2,
        #     padding=deconv_kernel // 2 - 1,
        # )
        # nn.init.kaiming_normal_(
        #     self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        # )
        # nn.init.constant_(self.kps_score_lowres.bias, 0)
        # self.up_scale = 2
        # self.out_channels = num_keypoints

        self.convs = []
        for i in range(self.num_convs):
            _in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            strides = 1
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Sequential(
                    Conv2d(
                        _in_channels, 
                        self.conv_out_channels, 
                        conv_kernel_size, 
                        strides, 
                        padding),
                    GroupNorm(32, self.conv_out_channels),
                    nn.ReLU(inplace=True)))
        self.convs = nn.Sequential(*self.convs)

        # self.convs1 = []
        # for i in range(self.num_convs):
        #     _in_channels = (
        #         self.in_channels if i == 0 else self.conv_out_channels)
        #     strides = 1
        #     padding = (conv_kernel_size1 - 1) // 2
        #     self.convs1.append(
        #         nn.Sequential(
        #             Conv2d(
        #                 _in_channels, 
        #                 self.conv_out_channels, 
        #                 conv_kernel_size1, 
        #                 strides, 
        #                 padding),
        #             GroupNorm(32, self.conv_out_channels),
        #             nn.ReLU(inplace=True)))
        # self.convs1 = nn.Sequential(*self.convs1)

        # self.convs2 = []
        # for i in range(self.num_convs):
        #     _in_channels = (
        #         self.in_channels if i == 0 else self.conv_out_channels)
        #     strides = 1
        #     padding = (conv_kernel_size1 - 1) // 2
        #     self.convs2.append(
        #         nn.Sequential(
        #             Conv2d(
        #                 _in_channels, 
        #                 self.conv_out_channels, 
        #                 conv_kernel_size1, 
        #                 strides, 
        #                 padding),
        #             GroupNorm(32, self.conv_out_channels),
        #             nn.ReLU(inplace=True)))
        # self.convs2 = nn.Sequential(*self.convs2)


        # self.updeconv1_1 = ConvTranspose2d(
        #     self.conv_out_channels,
        #     self.conv_out_channels // 2, 
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints // 2)
        # self.norm1 = GroupNorm(self.num_keypoints // 2, self.conv_out_channels // 2)
        # self.updeconv1_2 = ConvTranspose2d(
        #     self.conv_out_channels,
        #     self.conv_out_channels // 2, 
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints // 2)
        # self.norm2 = GroupNorm(self.num_keypoints // 2, self.conv_out_channels // 2)
        # self.updeconv2_1 = ConvTranspose2d(
        #     self.conv_out_channels // 2,
        #     self.num_keypoints // 2,
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints // 2)
        # self.updeconv2_2 = ConvTranspose2d(
        #     self.conv_out_channels // 2,
        #     self.num_keypoints // 2,
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints // 2)


        self.updeconv1_ = ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels, 
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.num_keypoints)
        self.norm1 = GroupNorm(self.num_keypoints, self.conv_out_channels)

        self.updeconv2_ = ConvTranspose2d(
            self.conv_out_channels,
            self.num_keypoints,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.num_keypoints)
        
#        self.conv_guide = Conv2d(
#            self.conv_out_channels, 
#            self.conv_out_channels, 
#            3, 
#            1, 
#            1)

#        self.dcn = DFConv2d_guide(self.conv_out_channels,
#            self.num_keypoints,
#            groups=self.num_keypoints)

        # self.norm2 = GroupNorm(self.num_keypoints, self.conv_out_channels)
        # self.final_conv = Conv2d(
        #                 self.conv_out_channels, 
        #                 self.num_keypoints, 
        #                 1, 
        #                 1, 
        #                 0,
        #                 groups=self.num_keypoints)
        # self.conv_offset = Conv2d(
        #                 self.conv_out_channels, 
        #                 self.num_keypoints * 2, 
        #                 1, 
        #                 1, 
        #                 0,
        #                 groups=self.num_keypoints)

        # self.convs_1 = []
        # for i in range(self.num_convs):
        #     _in_channels = (
        #         self.in_channels if i == 0 else self.conv_out_channels)
        #     strides = 1
        #     padding = (conv_kernel_size - 1) // 2
        #     self.convs_1.append(
        #         nn.Sequential(
        #             Conv2d(
        #                 _in_channels, 
        #                 self.conv_out_channels, 
        #                 conv_kernel_size, 
        #                 strides, 
        #                 padding),
        #             GroupNorm(36, self.conv_out_channels),
        #             nn.ReLU(inplace=True)))
        # self.convs_1 = nn.Sequential(*self.convs_1)

        # self.updeconv1_1 = ConvTranspose2d(
        #     self.conv_out_channels,
        #     self.conv_out_channels,
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints)
        # self.norm1_1 = GroupNorm(self.num_keypoints, self.conv_out_channels)
        # self.updeconv2_1 = ConvTranspose2d(
        #     self.conv_out_channels,
        #     self.num_keypoints,
        #     kernel_size=deconv_kernel_size,
        #     stride=2,
        #     padding=(deconv_kernel_size - 2) // 2,
        #     groups=self.num_keypoints)

        # #TODO 20201015
        # self.neighbor_points = []
        # grid_size = 3
        # for i in range(grid_size):  # i-th column
        #     for j in range(grid_size):  # j-th row
        #         neighbors = []
        #         if i > 0:  # left: (i - 1, j)
        #             neighbors.append((i - 1) * grid_size + j)
        #         if j > 0:  # up: (i, j - 1)
        #             neighbors.append(i * grid_size + j - 1)
        #         if j < grid_size - 1:  # down: (i, j + 1)
        #             neighbors.append(i * grid_size + j + 1)
        #         if i < grid_size - 1:  # right: (i + 1, j)
        #             neighbors.append((i + 1) * grid_size + j)
        #         self.neighbor_points.append(tuple(neighbors))

        # self.forder_trans = nn.ModuleList()  # first-order feature transition
        # self.sorder_trans = nn.ModuleList()  # second-order feature transition
        # for neighbors in self.neighbor_points:
        #     fo_trans = nn.ModuleList()
        #     so_trans = nn.ModuleList()
        #     for _ in range(len(neighbors)):
        #         # each transition module consists of a 5x5 depth-wise conv and
        #         # 1x1 conv.
        #         fo_trans.append(
        #             nn.Sequential(
        #                 Conv2d(
        #                     self.point_feat_channels,
        #                     self.point_feat_channels,
        #                     5,
        #                     stride=1,
        #                     padding=2,
        #                     groups=self.point_feat_channels),
        #                 Conv2d(self.point_feat_channels,
        #                           self.point_feat_channels, 1)))
        #         so_trans.append(
        #             nn.Sequential(
        #                 Conv2d(
        #                     self.point_feat_channels,
        #                     self.point_feat_channels,
        #                     5,
        #                     1,
        #                     2,
        #                     groups=self.point_feat_channels),
        #                 Conv2d(self.point_feat_channels,
        #                           self.point_feat_channels, 1)))
        #     self.forder_trans.append(fo_trans)
        #     self.sorder_trans.append(so_trans)


        # representation_size = 14 * 14 * 288
        # self.keypoints_weight = nn.Linear(representation_size, self.num_keypoints)
        # nn.init.normal_(self.cls_score.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
        # nn.init.constant_(self.final_conv.bias,-np.log(0.99/0.01))
        # nn.init.constant_(self.dcn.bias,-np.log(0.99/0.01))
        nn.init.constant_(self.updeconv2_.bias,-np.log(0.99/0.01))

    def forward(self, x):
        # heatmap = x.view(x.size(0), -1)
        # heatmap_weight = self.keypoints_weight(heatmap)
        x_ = self.convs(x)
        # x_2 = self.convs1(x)
        # x_ = torch.cat((x_1, x_2), 1)
        # c = self.point_feat_channels
        # # first-order fusion
        # x_fo = [None for _ in range(self.num_keypoints)]
        # for i, points in enumerate(self.neighbor_points):
        #     x_fo[i] = x[:, i * c:(i + 1) * c]
        #     for j, point_idx in enumerate(points):
        #         x_fo[i] = x_fo[i] + self.forder_trans[i][j](
        #             x[:, point_idx * c:(point_idx + 1) * c])

        # # second-order fusion
        # x_so = [None for _ in range(self.num_keypoints)]
        # for i, points in enumerate(self.neighbor_points):
        #     x_so[i] = x[:, i * c:(i + 1) * c]
        #     for j, point_idx in enumerate(points):
        #         x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])

        # # predicted heatmap with fused features
        # x = torch.cat(x_so, dim=1)
        x_ = self.updeconv1_(x_)
        x_ = F.relu(self.norm1(x_), inplace=True)     
        x_ = self.updeconv2_(x_)
      #  x_guide = self.conv_guide(x_)
      #  x_ = self.dcn(x_, x_guide)

        # x_1 = self.updeconv1_1(x_)
        # x_2 = self.updeconv1_2(x_)
        # x_1 = F.relu(self.norm1(x_1), inplace=True)
        # x_2 = F.relu(self.norm2(x_2), inplace=True)        
        # x_1 = self.updeconv2_1(x_1)
        # x_2 = self.updeconv2_2(x_2)
        # x_ = torch.cat((x_1, x_2), dim=1)
    
        # x_ = F.relu(self.norm2(x_), inplace=True)
        # x_heatmap = self.final_conv(x_)
        # x_offset = self.conv_offset(x_)
        # x_1 = self.convs_1(x)
        # x_1 = self.updeconv1_1(x_1)
        # x_1 = F.relu(self.norm1_1(x_1), inplace=True)
        # x_1 = self.updeconv2_1(x_1)

                
        return x_


def make_roi_keypoint_predictor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return func(cfg, in_channels)

