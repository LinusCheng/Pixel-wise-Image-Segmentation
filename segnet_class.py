import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision





class SegNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        inplace = True
        batchNorm_momentum = 0.1
        vgg16_bn = list(torchvision.models.vgg16_bn(pretrained = True).features.children())
        self.encoder1 = nn.Sequential(*vgg16_bn[0:6])
        self.encoder2 = nn.Sequential(*vgg16_bn[7:13])
        self.encoder3 = nn.Sequential(*vgg16_bn[14:23])
#        self.encoder4 = nn.Sequential(*vgg16_bn[24:33])
#        self.encoder5 = nn.Sequential(*vgg16_bn[34:-1])
#        self.decoder5 = nn.Sequential(
#                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(512, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(512, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(512, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      )
#        self.decoder4 = nn.Sequential(
#                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(512, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(512, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                      nn.BatchNorm2d(256, momentum=batchNorm_momentum),
#                      nn.ReLU(inplace),
#                      )
        self.decoder3 = nn.Sequential(
                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(256, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(256, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(128, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      )
        self.decoder2 = nn.Sequential(
                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(128, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      )
        self.decoder1 = nn.Sequential(
                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                      nn.ReLU(inplace),
                      nn.Conv2d(64, class_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      )
        
    def forward(self, x):
        x = self.encoder1(x)
        size1 = x.size()
        x, idx1 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x = self.encoder2(x)
        size2 = x.size()
        x, idx2 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x = self.encoder3(x)
        size3 = x.size()
        x, idx3 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
#        x = self.encoder4(x)
#        size4 = x.size()
#        x, idx4 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
#        x = self.encoder5(x)
#        size5 = x.size()
#        x, idx5 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
#        x = self.decoder5(F.max_unpool2d(x, idx5, kernel_size=(2, 2), stride=(2, 2), output_size = size5))
#        x = self.decoder4(F.max_unpool2d(x, idx4, kernel_size=(2, 2), stride=(2, 2), output_size = size4))

        x = self.decoder3(F.max_unpool2d(x, idx3, kernel_size=(2, 2), stride=(2, 2), output_size = size3))
        x = self.decoder2(F.max_unpool2d(x, idx2, kernel_size=(2, 2), stride=(2, 2), output_size = size2))
        x = self.decoder1(F.max_unpool2d(x, idx1, kernel_size=(2, 2), stride=(2, 2), output_size = size1))
        x = F.softmax(x, dim=1)
        return x