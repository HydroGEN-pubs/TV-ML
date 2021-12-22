import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import Shapes as shp
#from Shapes import conv2d_shape
#from Shapes import pool2d_shape


from torch.nn.utils.rnn import pack_padded_sequence
# %%
## RMM  NN
#Define  the model
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[25, 25],  channels=2, verbose=False):
        super(RMM_NN, self).__init__()
        self.use_dropout = False
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Inputs Parameters
        # ---------------------------------------------------------------------
        Cin = channels
        Hin = grid_size[0]
        Win = grid_size[1]
        in_shape = [Cin, Hin, Win]

        # Convolution Layer 1 parameters
        Coutfinal = 1
        Cout1 = 8
        Cout2 = 16
        Cout3 = 32
        cnv_kernel_size = 3
        cnv_kernel_size2 = 3
        cnv_stride2 = 1
        cnv_stride = 1
        cnv_padding = 1  # verify that it satisfies equation
        cnv_dilation = 1

        # Pooling Layer parameters
        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0
        pool_dilation = 1

        # ---------------------------------------------------------------------
        #  definitions
        # ---------------------------------------------------------------------
        self.contract1 = nn.Sequential(
            nn.Conv2d(
                in_channels= Cin,
                out_channels=Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=Cout1,
                out_channels=Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding).float()
        )

        self.contract2 = nn.Sequential(
            nn.Conv2d(
                in_channels= Cout1,
                out_channels=Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=Cout1,
                out_channels=Cout2,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding).float()
        )

        self.contract3 = nn.Sequential(
            nn.Conv2d(
                in_channels= Cout2,
                out_channels=Cout2,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=Cout2,
                out_channels=Cout3,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding).float()
        )

        self.expand3 = nn.Sequential(
             nn.Conv2d(
                 in_channels=Cout3,
                 out_channels=Cout3,
                 kernel_size=cnv_kernel_size,
                 stride=cnv_stride,
                 padding=cnv_padding
             ).float(),
             #torch.nn.BatchNorm2d(out_channels=Cout),
             nn.ReLU(),
             nn.Conv2d(
                 in_channels=Cout3,
                 out_channels=Cout3,
                 kernel_size=cnv_kernel_size,
                 stride=cnv_stride,
                 padding=cnv_padding
             ).float(),
             #torch.nn.BatchNorm2d(out_channels=Cout),
             nn.ReLU(),
             nn.ConvTranspose2d(
                 in_channels=Cout3,
                 out_channels=Cout2,
                 kernel_size=pool_kernel_size,
                 stride=pool_stride,
                 padding=pool_padding,
                 output_padding=0).float()
         )

        self.expand2 = nn.Sequential(
            nn.Conv2d(
                in_channels=2*Cout2,
                out_channels=2*Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2*Cout1,
                out_channels=2*Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=2*Cout1,
                out_channels=Cout1,
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
                output_padding=0).float()
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2*Cout1,
                out_channels=2*Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2*Cout1,
                out_channels=2*Cout1,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #torch.nn.BatchNorm2d(out_channels=Cout),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=2*Cout1,
                out_channels=Coutfinal,
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
                output_padding=1).float()
        )
        # set input elements zero with probability p = 0.5 (default)
        #self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c2d_s1 = shp.conv2d_shape(input_shape=in_shape, cout=Cout1, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        c2d_s2 = shp.conv2d_shape(input_shape=c2d_s1, cout=Cout1, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        pool2d_s1 = shp.pool2d_shape(input_shape=c2d_s1, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)
        c2d_s3 = shp.conv2d_shape(input_shape=pool2d_s1, cout=Cout2, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        c2d_s4 = shp.conv2d_shape(input_shape=c2d_s3, cout=Cout2, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        pool2d_s2 = shp.pool2d_shape(input_shape=c2d_s4, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)
        trans2d_s1 = shp.convTranspose2d_shape(input_shape=pool2d_s2, cout=Cout1, padding=cnv_padding,
                            kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)


        # self.conv1 = self.contract(in_channels, 32, 7, 3)
        # self.conv2 = self.contract(32, 64, 3, 1)
        # self.conv3 = self.contract(64, 128, 3, 1)

        # self.upconv3 = self.expand(128, 64, 3, 1)
        # self.upconv2 = self.expand(64*2, 32, 3, 1)
        # self.upconv1 = self.expand(32*2, 1, 3, 1)
        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        #L_Fin1 = int(np.prod(np.floor(trans2d_s2)))
        #L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        #L_Fout1 = 700  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        #L_Fout1 = 7000  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        #L_Fout1 = 20000  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000

        #self.dense = nn.Linear(L_Fin1, L_Fout1).float()
        #self.dense = nn.Bilinear(L_Fin1, L_Fout1).float()

        #L_Fin2 = L_Fout1
        #L_Fout2 = int(Hin*Win)
        #self.out = nn.Linear(L_Fin2, L_Fout2).float()

        # ---------------------------------------------------------------------
        # Print Expected shapes
        # ---------------------------------------------------------------------
        # if verbose:
        #     print("-- Model shapes --")
        #     print('Input Shape:', in_shape)
        #     print('Expected C2D Shape1:', c2d_s1)
        #     print('Expected Pool Shape:', pool2d_s1)
        #     print('Linear 1', L_Fin1, L_Fout1)
        #     print('Linear 2', L_Fin2, L_Fout2)

    def forward(self, x):
        conv1 = self.contract1(x)
        if self.verbose: print('contract1 ', conv1.shape)
        conv2 = self.contract2(conv1)
        if self.verbose: print('contract2 ', conv2.shape)

        conv3 = self.contract3(conv2)
        if self.verbose: print('contract3 ', conv3.shape)

        upconv3 = self.expand3(conv3)
        if self.verbose: print('expand3 ', upconv3.shape)
        upconv2 = self.expand2(torch.cat([upconv3, conv2], 1))
        if self.verbose: print('expand2 ', upconv2.shape)
        out = self.expand1(torch.cat([upconv2, conv1], 1))   #upconv2
        if self.verbose: print('expand1 ', out.shape)

        #layer0 = self.layer0(x)
        #if self.verbose: print('Step 1: Layer 1', out.shape)
        #layer1 = self.layer1(layer0)
        #if self.verbose: print('Step 2: Reshape', out.shape)
    #   if self.use_dropout:
    #       out = self.drop_out(out)
    #       if self.verbose: print('Step 2b: Dropout', out.shape)
    #   out = self.dense(out)
    #   if self.verbose: print('Step 3: Dense', out.shape)
    #   out = self.out(out)
    #   if self.verbose: print('Step 4: Linear Out', out.shape)
        out = out.reshape(out.size(0), -1)
        return out        

'''
class RMM_NN2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand
        '''