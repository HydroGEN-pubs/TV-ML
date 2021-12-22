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
#First NN model from Seb - Step 2
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[100, 50], channels=2):
        super(FirstNeuroNet, self).__init__()
        self.use_dropout = True
        # Input as [N,Cin,Hin,Win] tensor
        #
        C2D_Cin = channels
        C2D_Hin = grid_size[1]
        C2D_Win = grid_size[0]
        #
        # Select
        C2D_Cout = 16
        C2D_kernel_size = 3
        C2D_stride = 1
        C2D_padding = 1  # verify that it satisfies equation
        C2D_dilation = 1
        #
        # Calulate
        C2D_Hout = (C2D_Hin + 2*C2D_padding - C2D_dilation *
                    (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        C2D_Wout = (C2D_Win + 2*C2D_padding - C2D_dilation *
                    (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        # ---------------------------------------------------------------------
        # ReLU() for N = batch_size
        # [N,*]->[N,*] no change in shape
        # ---------------------------------------------------------------------
        #
        # ---------------------------------------------------------------------
        # MaxPool2d(kernel_size,stride,...) for N = batch_size
        # [N,C,Hin,Win]->[N,C,Hout,Wout]
        # kernel_size = 2 - Select
        # stride = 2 (default kernel_size)
        # dialation = 1 (default)
        # padding = 1 (default)
        #
        # Hout = [Hin + 2xpadding[0] - dilation[0]x(kernel_size[0] - 1) - 1]/stride[0] + 1
        #
        # Wout = [Win + 2xpadding[1] - dilation[1]x(kernel_size[1] - 1) - 1]/stride[1] + 1
        # ---------------------------------------------------------------------
        #
        # Input as [N,C,Hin,Win] tensor
        #
        MP2D_C = C2D_Cout
        MP2D_Hin = C2D_Hin
        MP2D_Win = C2D_Win
        #
        # Select
        MP2D_kernel_size = 2
        MP2D_stride = 2
        MP2D_padding = 0
        MP2D_dilation = 1
        #
        # Calulate
        MP2D_Hout = (MP2D_Hin + 2*MP2D_padding - MP2D_dilation *
                     (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        MP2D_Wout = (MP2D_Win + 2*MP2D_padding - MP2D_dilation *
                     (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=C2D_Cin,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.ReLU() ,
            nn.MaxPool2d(
                kernel_size=MP2D_kernel_size,
                stride=MP2D_stride
            )
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()
        # ---------------------------------------------------------------------
        # Linear(Fin,Fout) for N = batch_size
        # [N,Fin]->[N,Fout] - Input tensor is flattened using reshape or view
        #
        # Fin = CxHinxWin
        # Fout = 0.5xFin
        # ---------------------------------------------------------------------
        L_C = MP2D_C
        L_Hin = MP2D_Hout
        L_Win = MP2D_Wout
        L_Fin = int(L_C*L_Hin*L_Win)
        L_Fout = int(round(0.5*L_Fin))
        self.dense = nn.Linear(L_Fin, L_Fout).float()
        L_Fin = L_Fout
        L_Fout = int(C2D_Hin*C2D_Win)
        self.out = nn.Linear(L_Fin, L_Fout).float()

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        if self.use_dropout:
            out = self.drop_out(out)
        out = self.dense(out)
        out = self.out(out)
        return out


# %%
#First NN model from Seb - Step 2 Modified to use shape calcs
class FirstNeuroNetShapes(nn.Module):
    def __init__(self, grid_size=[50,100], channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, MP2D_stride=2):
        super(FirstNeuroNetShapes, self).__init__()
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
    
        # Convolution Layer parameters
        C2D_Cout = 16
        #C2D_kernel_size = 3
        C2D_stride = 1
        C2D_padding = 1  # verify that it satisfies equation
        C2D_dilation = 1

        # Pooling layer parameters
        #MP2D_kernel_size = 2
        MP2D_padding = 0
        MP2D_dilation = 1
        #MP2D_stride = 1

        # ---------------------------------------------------------------------
        # Layer 1 definition
        # ---------------------------------------------------------------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=Cin,
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

        pool2d_s1 = shp.pool2d_shape(input_shape=c2d_s1, padding=MP2D_padding,
                                 kernel_size=MP2D_kernel_size, dilation=MP2D_dilation,
                                 stride=MP2D_stride)

        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool2d_s1)))
        L_Fout1 = int(round(0.5*L_Fin1))
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
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Linear Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out


# %%
## RMM NN
#Define  the model
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[100, 50],  channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, C2D_kernel_size2=3,
                 MP2D_stride=2):
        super(RMM_NN, self).__init__()
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
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size2,
                stride=C2D_stride2,
                padding=C2D_padding
            ).float(),
            nn.ReLU(),
              nn.Conv2d(
                in_channels=C2D_Cout,
                out_channels=C2D_Cout,
                kernel_size=C2D_kernel_size,
                stride=C2D_stride,
                padding=C2D_padding
            ).float(),
            nn.ReLU(),
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

        c2d_s3 = shp.conv2d_shape(input_shape=c2d_s2, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        c2d_s4 = shp.conv2d_shape(input_shape=c2d_s3, cout=C2D_Cout, padding=C2D_padding,
                              kernel_size=C2D_kernel_size, dilation=C2D_dilation, stride=C2D_stride)
        
        pool2d_s1 = shp.pool2d_shape(input_shape=c2d_s4, padding=MP2D_padding, kernel_size=MP2D_kernel_size,
                                 dilation=MP2D_dilation, stride=MP2D_stride)

        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool2d_s1))) 
        L_Fout1 = 750  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
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
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out

# %%
## copy of NN w/ BC from Seb's
class RMM_NN2(nn.Module):
    def __init__(self, grid_size=[100, 50], channels=1):
        super(RMM_NN2, self).__init__()
        g_size = reduce(lambda a, b: a * b, grid_size, 1)
        w_in = g_size * channels
        w_out = g_size
        padding = 1
        dense_layer_size = 10000
        self.left = 30
        self.right = 30
        self.use_dropout = True
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=padding
            ).float(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.drop_out = nn.Dropout()
        # Size +2 to accomodate left/right boundary
        self.dense = nn.Linear(16*50*25 , dense_layer_size).float()
        self.out = nn.Linear(dense_layer_size, w_out).float()

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        if self.use_dropout:
            out = self.drop_out(out)
        # Add left/right boundary
        batch_size = out.shape[0]
        boundaries = []
        for i in range(batch_size):
            boundaries.append(self.left)
            boundaries.append(self.right)

        boundaries = torch.tensor(boundaries, dtype=torch.float32).to(DEVICE)
        boundaries = boundaries.reshape((batch_size, -1))

        out = torch.cat((out, boundaries), 1)
        out = self.dense(out)
        out = self.out(out)
        return out

# %%
## RMM 3D NN
#Define  the model
class RMM_NN_3D(nn.Module):
    def __init__(self, grid_size=[100, 1, 50],  channels=2, verbose=False,):
        super(RMM_NN_3D, self).__init__()
        self.use_dropout = True
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Inputs Parameters
        # ---------------------------------------------------------------------
        Cin = channels
        Din = grid_size[0]
        Hin = grid_size[1]
        Win = grid_size[2]
        in_shape = [Cin, Din, Hin, Win]

        # Convolution Layer 1 parameters
        Cout = 16
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
            nn.Conv3d(
                in_channels= Cin,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            nn.ReLU(),
            #  nn.Conv3d(
            #     in_channels=Cout,
            #     out_channels=Cout,
            #     kernel_size=conv_kernel_size,
            #     stride=conv_stride,
            #     padding=conv_padding
            # ).float(),
            # #nn.LeakyReLU(),
            # nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            )
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c3d_s1 = shp.conv3d_shape(input_shape=in_shape, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)

        pool3d_s1 = shp.pool3d_shape(input_shape=c3d_s1, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)


        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool3d_s1)))
        #L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        L_Fout1 = 750  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        self.dense = nn.Linear(L_Fin1, L_Fout1).float()

        L_Fin2 = L_Fout1
        L_Fout2 = int(Hin*Win*Din)
        self.out = nn.Linear(L_Fin2, L_Fout2).float()

        # ---------------------------------------------------------------------
        # Print Expected shapes
        # ---------------------------------------------------------------------
        if verbose:
            print("-- Model shapes --")
            print('Input Shape:', in_shape)
            print('Expected C3D Shape1:', c3d_s1)
            print('Expected Pool Shape:', pool3d_s1)
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




## RMM 3D CONV LSTM  pytorch_convolutional_rnn from https://github.com/kamo-naoyuki/pytorch_convolutional_rnn
##
#Define  the model
class RMM_CNNLSTM_3D(nn.Module):
    def __init__(self, grid_size=[100,1, 50], channels=2):
        super(RMM_CNNLSTM_3D, self).__init__()
        self.use_dropout = True
        # Input as [N,Cin,Hin,Win] tensor
        #
        C2D_Cin = channels
        C2D_Hin = grid_size[1]
        C2D_Win = grid_size[0]
        C2D_Din = grid_size[2]
        #
        # Select
        C2D_Cout = 16
        C2D_kernel_size = 3
        C2D_kernel_size2 = 3
        C2D_stride2 = 1
        C2D_stride = 1
        C2D_padding = 1  # verify that it satisfies equation
        C2D_dilation = 1
        #
        # Calculate
        C2D_Dout = (C2D_Din + 2*C2D_padding - C2D_dilation *
                    (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        C2D_Hout = (C2D_Hin + 2*C2D_padding - C2D_dilation *
                    (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        C2D_Wout = (C2D_Win + 2*C2D_padding - C2D_dilation *
                    (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        # ---------------------------------------------------------------------
        # ReLU() for N = batch_size
        # [N,*]->[N,*] no change in shape
        # ---------------------------------------------------------------------
        #
        # ---------------------------------------------------------------------
        # MaxPool2d(kernel_size,stride,...) for N = batch_size
        # [N,C,Hin,Win]->[N,C,Hout,Wout]
        # kernel_size = 2 - Select
        # stride = 2 (default kernel_size)
        # dialation = 1 (default)
        # padding = 1 (default)
        #
        # Hout = [Hin + 2xpadding[0] - dilation[0]x(kernel_size[0] - 1) - 1]/stride[0] + 1
        #
        # Wout = [Win + 2xpadding[1] - dilation[1]x(kernel_size[1] - 1) - 1]/stride[1] + 1
        # ---------------------------------------------------------------------
        #
        # Input as [N,C,Hin,Win] tensor
        #
        MP2D_C = C2D_Cout
        MP2D_Hin = C2D_Hout
        MP2D_Win = C2D_Wout
        MP2D_Din = C2D_Dout
        #
        # Select
        MP2D_kernel_size = 2
        
        MP2D_stride = 2
        MP2D_padding = 0
        MP2D_dilation = 1
        #
        # Calulate
        MP2D_Dout = (MP2D_Din + 2*MP2D_padding - MP2D_dilation *
                     (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        MP2D_Hout = (MP2D_Hin + 2*MP2D_padding - MP2D_dilation *
                     (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        MP2D_Wout = (MP2D_Win + 2*MP2D_padding - MP2D_dilation *
                     (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        #C2D_Hout = (MP2D_Hout + 2*C2D_padding - C2D_dilation *
        #             (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        #C2D_Wout = (MP2D_Wout + 2*C2D_padding - C2D_dilation *
        #             (C2D_kernel_size - 1) - 1)/C2D_stride + 1
        #MP2D_Hout2 = (MP2D_Hout + 2*MP2D_padding - MP2D_dilation *
        #             (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        #MP2D_Wout2 = (MP2D_Wout + 2*MP2D_padding - MP2D_dilation *
        #             (MP2D_kernel_size - 1) - 1)/MP2D_stride + 1
        # layers of CONV2D and ReLU.  I wish I could figure out how to pool in between the Conv2D and ReLU steps but the resulting
        # Tensor size is always wrong
        self.layer1 = nn.Sequential(
            convolutional_rnn.Conv3dGRU(in_channels=C2D_Cin,  # Corresponds to input size
                out_channels=C2D_Cout,  # Corresponds to hidden size
                kernel_size=C2D_kernel_size,  # Int or List[int]
                num_layers=2,
                bidirectional=False,
                dilation=2, stride=C2D_stride, dropout=0.5,length=1,
                batch_first=True).float(),
            #nn.LeakyReLU(),
            nn.ReLU(),
            #  nn.Conv3d(
            #     in_channels=C2D_Cout,
            #     out_channels=C2D_Cout,
            #     kernel_size=C2D_kernel_size,
            #     stride=C2D_stride,
            #     padding=C2D_padding
            # ).float(),
            # #nn.LeakyReLU(),
            # nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=MP2D_kernel_size,
                stride=MP2D_stride
            )
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()
        # ---------------------------------------------------------------------
        # Linear(Fin,Fout) for N = batch_size
        # [N,Fin]->[N,Fout] - Input tensor is flattened using reshape or view
        #
        # Fin = CxHinxWin
        # Fout = 0.5xFin
        # ---------------------------------------------------------------------
        L_C = MP2D_C
        L_Hin = MP2D_Hout
        L_Win = MP2D_Wout
        L_Din = MP2D_Dout
        #L_Hin = C2D_Hin
        #L_Win = C2D_Win
        L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        #print(L_Fin, L_C,L_Hin,L_Win,L_Din)
        #L_Fin = out.size(0)
        #print(L_Fin)
        L_Fout = int(round(0.5*L_Fin))
        L_Fout = 750  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        self.dense = nn.Linear(L_Fin, L_Fout).float()
        L_Fin = L_Fout
        L_Fout = int(C2D_Hin*C2D_Win*C2D_Din)
        self.out = nn.Linear(L_Fin, L_Fout).float()

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        if self.use_dropout:
            out = self.drop_out(out)
        out = self.dense(out)
        out = self.out(out)
        return out        
# %%
## RMM NN2
#Define  the model
class RMM_NN2(nn.Module):
    def __init__(self, grid_size=[100, 50],  channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, C2D_kernel_size2=3,
                 MP2D_stride=2):
        super(RMM_NN2, self).__init__()
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
            #nn.LeakyReLU(),
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
        L_Fin1 = int(np.prod(np.floor(pool2d_s3))) 
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
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out
# %%
## RMM NN2
#Define  the model
class RMM_NN2_withBCDT(nn.Module):
    def __init__(self, grid_size=[100, 50],  channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, C2D_kernel_size2=3,
                 MP2D_stride=2):
        super(RMM_NN2_withBCDT, self).__init__()
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
            #nn.LeakyReLU(),
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
        L_Fin1 = int(np.prod(np.floor(pool2d_s3))) #+ 3
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

        boundaries = torch.tensor(boundaries, dtype=torch.float32)
        boundaries = boundaries.reshape((batch_size, -1))

        #out = torch.cat((out, boundaries), 1)
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out

# %%
## RMM NN4
#Define  the model
class RMM_NN_4(nn.Module):
    def __init__(self, grid_size=[100, 50],  channels=2, verbose=False,
                 C2D_kernel_size=3, MP2D_kernel_size=2, C2D_kernel_size2=3,
                 MP2D_stride=2):
        super(RMM_NN_4, self).__init__()
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
            #nn.LeakyReLU(),
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

# %%
# %%
## RMM 2D 3D NN
#Define  the model
class RMM_NN_2D3D(nn.Module):
    def __init__(self, grid_size=[100, 1, 50],  channels=2, verbose=False,):
        super(RMM_NN_2D3D, self).__init__()
        self.use_dropout = True
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Inputs Parameters
        # ---------------------------------------------------------------------
        Cin = channels
        Din = grid_size[0]
        Hin = grid_size[1]
        Win = grid_size[2]
        in_shape = [Cin, Din, Hin, Win]

        # Convolution Layer 1 parameters
        Cout = 16
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
            nn.Conv3d(
                in_channels= Cin,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            nn.ReLU(),
            #  nn.Conv3d(
            #     in_channels=Cout,
            #     out_channels=Cout,
            #     kernel_size=conv_kernel_size,
            #     stride=conv_stride,
            #     padding=conv_padding
            # ).float(),
            # #nn.LeakyReLU(),
            # nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            )
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c3d_s1 = shp.conv3d_shape(input_shape=in_shape, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)

        pool3d_s1 = shp.pool3d_shape(input_shape=c3d_s1, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)


        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool3d_s1)))
        #L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        L_Fout1 = 750  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        self.dense = nn.Linear(L_Fin1, L_Fout1).float()

        L_Fin2 = L_Fout1
        L_Fout2 = int(Hin*Win*Din)
        self.out = nn.Linear(L_Fin2, L_Fout2).float()

        # ---------------------------------------------------------------------
        # Print Expected shapes
        # ---------------------------------------------------------------------
        if verbose:
            print("-- Model shapes --")
            print('Input Shape:', in_shape)
            print('Expected C3D Shape1:', c3d_s1)
            print('Expected Pool Shape:', pool3d_s1)
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


