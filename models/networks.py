import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import make_layers, cfgs
from torchvision.models import resnet50


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(6, 64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out      # 64


class AnalogyRelationNetwork(nn.Module):
    def __init__(self, n_classes, vgg=True):
        super(AnalogyRelationNetwork, self).__init__()
        if vgg:
            self.feature = make_layers(cfgs['E'], batch_norm=True)
            FEAT_DIM = 512
            self.size_fc = 1
        else:
            self.feature = CNNEncoder()
            FEAT_DIM = 64
            self.size_fc = 6
        self.layer1 = nn.Sequential(
                        nn.Conv2d(2 * FEAT_DIM, FEAT_DIM, kernel_size=3, padding=1),
                        nn.BatchNorm2d(FEAT_DIM, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(FEAT_DIM, int(FEAT_DIM / 2), kernel_size=3, padding=1),
                        nn.BatchNorm2d(int(FEAT_DIM / 2), momentum=1, affine=True),
                        nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(int(FEAT_DIM / 2), 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.FEAT_DIM = FEAT_DIM

    def forward(self, data, candidates):
        anchor_feature = self.feature(data)
        B, n_can, C, H, W = candidates.shape
        candidate_features = self.feature(candidates.view(-1, C, H, W)).view(B, n_can, self.FEAT_DIM, anchor_feature.shape[-2], anchor_feature.shape[-1])
        anchor_feature = anchor_feature.unsqueeze(1).repeat(1, n_can, 1, 1, 1)
        assert anchor_feature.shape == candidate_features.shape
        x = torch.cat((candidate_features, anchor_feature), dim=2)      # N, num_candidates, 2 x FEAT_DIM, 8, 8 for input size 256 x 256
        x = x.flatten(0, 1)

        # relation module
        out = self.layer1(x)
        out = self.avgpool(self.layer2(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.reshape(B, n_can)
        return out


class AnalogySiamese(nn.Module):
    def __init__(self, feature_dim, method='concat'):
        super(AnalogySiamese, self).__init__()
        self.embedding = resnet50(num_classes=256)      # C = 256
        self.feature_dim = feature_dim
        self.method = method
        combined_feat_dim = 2 * 256 if method == 'concat' else 256

        self.fc = nn.Sequential(
            nn.Linear(combined_feat_dim, combined_feat_dim),
            nn.ReLU(True),
            nn.Linear(combined_feat_dim, feature_dim),
        )

    def forward(self, x):
        # input is A, B, C, 8xD
        separate = False
        if x.ndim == 5:
            separate = True
            batch_size, total, _, _, _ = x.shape
            x = x.flatten(0, 1)
        x = self.embedding(x)       # B x 11, C
        x = x.reshape((batch_size, total, ) + x.shape[1:])         # B, 11, C
        x_A = x[:, 0].unsqueeze(1)       # B, 1, C
        x_B = x[:, 1].unsqueeze(1)       # B, 1, C
        x_C = x[:, 2].unsqueeze(1)       # B, 1, C
        x_D = x[:, 3:]                   # B, 8, C
        if self.method == 'concat':
            x_AB = torch.cat((x_A, x_B), dim=-1)                               # B, 1, 2C
            x_CD = torch.cat((x_C.repeat(1, total - 3, 1), x_D), dim=-1)       # B, 8, 2C
        elif self.method == 'minus':
            x_AB = x_A - x_B                                 # B, 1, C
            x_CD = x_C.repeat(1, total - 3, 1) - x_D         # B, 8, C
        x = torch.cat((x_AB, x_CD), dim=1)                   # B, 9, 2C
        x = x.flatten(0, 1)                                  # B x 9, 2C
        x = self.fc(x)                                       # B x 9, feature_dim

        # l2 norm of x
        xn = torch.norm(x, p=2, dim=1).detach().unsqueeze(1)
        x = x.div(xn.expand_as(x))
        if separate:
            x = x.reshape((batch_size, total - 2, self.feature_dim))
        return x


class AnalogySiameseVGG(nn.Module):
    def __init__(self, feature_dim=256, method='concat'):
        super(AnalogySiameseVGG, self).__init__()
        self.feature = make_layers(cfgs['E'], batch_norm=False, in_channels=3)
        self.FEAT_DIM = 512
        self.feature_dim = feature_dim
        self.method = method

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))             # for image
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))            # for patch

        # for part-whole relation classification
        combined_feat_dim = 2 * self.FEAT_DIM if method == 'concat' else self.FEAT_DIM
        self.conv = nn.Sequential(
                        nn.Conv2d(combined_feat_dim, self.FEAT_DIM, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(self.FEAT_DIM, momentum=1, affine=True),
                        nn.ReLU())
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(self.FEAT_DIM, self.FEAT_DIM),
            nn.ReLU(True),
            nn.Linear(self.FEAT_DIM, self.FEAT_DIM // 2),
            nn.ReLU(True),
            nn.Linear(self.FEAT_DIM // 2, feature_dim)
        )

    def forward(self, x):
        # input is A, B, C, 8xD
        assert x.ndim == 5
        batch_size, total, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.feature(x)       # B x 11, C, H, W
        x = x.reshape((-1, total, ) + x.shape[1:])         # B, 11, C, H, W
        x_A = self.avgpool(x[:, 0])      # B, C, 4, 4
        x_B = self.avgpool1(x[:, 1])     # B, C, 1, 1
        x_C = self.avgpool(x[:, 2])      # B, C, 4, 4
        x_D = self.avgpool1(x[:, 3:].flatten(0, 1)).reshape((batch_size, total - 3, self.FEAT_DIM, 1, 1))      # B, 8, C, 1, 1

        x_B = x_B.repeat(1, 1, x_A.shape[-2], x_A.shape[-1])       # B, C, 4, 4
        x_D = x_D.repeat(1, 1, 1, x_C.shape[-2], x_C.shape[-1])  # B, 8, C, 4, 4
        if self.method == 'concat':
            x_AB = torch.cat((x_A, x_B), dim=1).unsqueeze(1)           # B, 1, 2C, 4, 4
            x_CD = torch.cat((x_C.repeat(1, total - 3, 1, 1, 1), x_D), dim=2)  # B, 8, 2C, 4, 4
        elif self.method == 'minus':
            x_AB = (x_A - x_B).unsqueeze(1)                  # B, 1, C, 4, 4
            x_CD = x_C.repeat(1, total - 3, 1, 1, 1) - x_D           # B, 8, C, 4, 4
        x = torch.cat((x_AB, x_CD), dim=1)                   # B, 9, 2C, 4, 4
        x = self.maxpool(self.conv(x.flatten(0, 1)))         # B x 9, C, 1, 1
        x = torch.flatten(x, 1)                              # B x 9, C
        x = self.embedding(x)                                # B x 9, feature_dim
        # l2 norm of x
        xn = torch.norm(x, p=2, dim=1).detach().unsqueeze(1)
        x = x.div(xn.expand_as(x))
        x = x.reshape((batch_size, total - 2, self.feature_dim))
        return x
