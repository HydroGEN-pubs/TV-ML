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
## RMM NN4
#Define  the model
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[100, 50],  channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, C2D_kernel_size2=3,
                 MP2D_stride=2):
        super(RMM_NN_5, self).__init__()
        self.use_dropout = True
        self.verbose = verbose 
        # ---------------------------------------------------------------------
        # Inputs Parameters
        # ---------------------------------------------------------------------
        # Input as [N,Cin,Hin,Win] tensor
        Cin = channels
        Hin = grid_size[0]
        Win = grid_size[1]
        in_shape = [Cin, Hin, Win]
        
        # Convolution Layer 1 parameters
        C2D_Cout = 16
        #C2D_kernel_size = 3
        C2D_stride = 1
        C2D_padding = 1  # verify that it satisfies equation
        C2D_dilation = 1

        # Convolution Layer 2 parameters
        #C2D_kernel_size2 = 3
        C2D_stride2 = 1

        #Pooling Layer parameters
        #MP2D_kernel_size = 2
        #MP2D_stride = 2
        MP2D_padding = 0
        MP2D_dilation = 1
        ## BC and Time Step info
        self.LeftBC = 10
        self.RightBC = 10
        self.DeltaT = 1

        # ---------------------------------------------------------------------
        # Layer 1 definition 
        # ---------------------------------------------------------------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size2,
                stride=C2D_stride2,
                padding=C2D_padding
            ).float(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=MP2D_kernel_size,
                stride=MP2D_stride
            ),
              nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=MP2D_kernel_size,
                stride=MP2D_stride
            ),
            nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=MP2D_kernel_size,
                stride=MP2D_stride
            )
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c2d_s1 = shp.conv2d_shape(input_shape=in_shape, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)

        c2d_s2 = shp.conv2d_shape(input_shape=c2d_s1, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size2, dilation=C2D_dilation, stride=C2D_stride)
        pool2d_s1 = shp.pool2d_shape(input_shape=c2d_s2, padding=MP2D_padding, kernel_size=MP2D_kernel_size,
                                 dilation=MP2D_dilation, stride=MP2D_stride)

        c2d_s3 = shp.conv2d_shape(input_shape=pool2d_s1, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        c2d_s4 = shp.conv2d_shape(input_shape=c2d_s3, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        pool2d_s2 = shp.pool2d_shape(input_shape=c2d_s4, padding=MP2D_padding, kernel_size=MP2D_kernel_size,
                                 dilation=MP2D_dilation, stride=MP2D_stride)
        c2d_s5 = shp.conv2d_shape(input_shape=pool2d_s2, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        c2d_s6 = shp.conv2d_shape(input_shape=c2d_s5, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        pool2d_s3 = shp.pool2d_shape(input_shape=c2d_s6, padding=MP2D_padding, kernel_size=MP2D_kernel_size,
                                 dilation=MP2D_dilation, stride=MP2D_stride)

        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool2d_s3))) + 3
        L_Fout1 = 2000  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        self.dense = nn.Linear(L_Fin1, L_Fout1).float()

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
            print('Expected C2D Shape2:', c2d_s2)
            print('Expected C2D Shape3:', c2d_s3)
            print('Expected C2D Shape4:', c2d_s4)
            print('Expected Pool Shape:', pool2d_s1)
            print('Linear 1', L_Fin1, L_Fout1)
            print('Linear 2', L_Fin2, L_Fout2)
    
    # ---------------------------------------------------------------------
    # Forward Model
    # ---------------------------------------------------------------------
    def forward(self, x):
        out = self.layer1(x)
        if self.verbose: print('Step 1: Layer 1', out.shape)
        out = out.reshape(out.size(0), -1)
       # Add left/right boundary
        batch_size = out.shape[0]
        boundaries = []
        for i in range(batch_size):
            boundaries.append(self.LeftBC)
            boundaries.append(self.RightBC)
            boundaries.append(self.DeltaT)
        #print("BC DT", self.RightBC, self.LeftBC, self.DeltaT)

        boundaries = torch.tensor(boundaries, dtype=torch.float32)
        boundaries = boundaries.reshape((batch_size, -1))

        out = torch.cat((out, boundaries), 1)
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out
