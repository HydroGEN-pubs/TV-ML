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

#Add Sand Tank path to the sys path
#sys.path.append('/Users/reed/Projects/HydroFrame-ML/pytorch_convolutional_rnn')
#import convolutional_rnn

from torch.nn.utils.rnn import pack_padded_sequence
# %%
## RMM 3D NN
#Define  the model
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[25, 25],  channels=2, verbose=False,):
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
        Cout = 5
        #Cout = 18
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
        # Layer 1 definition 
        # ---------------------------------------------------------------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels= Cin,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            #nn.LogSoftmax(),
            nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=Cout,
            #     out_channels=Cout,
            #     kernel_size=cnv_kernel_size,
            #     stride=cnv_stride,
            #     padding=cnv_padding
            # ).float(),
            # nn.ReLU(),
            # #nn.Softplus(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            ),
            nn.Conv2d(
                in_channels=Cout,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride)
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c2d_s1 = shp.conv2d_shape(input_shape=in_shape, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        c2d_s2 = shp.conv2d_shape(input_shape=c2d_s1, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)

        pool2d_s1 = shp.pool2d_shape(input_shape=c2d_s1, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)
        c2d_s3 = shp.conv2d_shape(input_shape=pool2d_s1, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        pool2d_s2 = shp.pool2d_shape(input_shape=c2d_s3, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)


        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool2d_s2)))
        #L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        L_Fout1 = 100  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        #L_Fout1 = 7000  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        #L_Fout1 = 20000  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000

        self.dense = nn.Linear(L_Fin1, L_Fout1).float()
        #self.dense = nn.Bilinear(L_Fin1, L_Fout1).float()

        L_Fin2 = L_Fout1
        L_Fout2 = int(Hin*Win)
        self.out = nn.Linear(L_Fin2, L_Fout2).float()

        # ---------------------------------------------------------------------
        # Print Expected shapes
        # ---------------------------------------------------------------------
        if verbose:
            print("-- Model shapes --")
            print('Input Shape:', in_shape)
            print('Expected C2D Shape1:', c2d_s1)
            print('Expected Pool Shape:', pool2d_s1)
            print('Linear 1', L_Fin1, L_Fout1)
            print('Linear 2', L_Fin2, L_Fout2)

    def forward(self, x):
        out = self.layer1(x)
        if self.verbose: print('Step 1: Layer 1', out.shape)
        out = out.reshape(out.size(0), -1)
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out        

