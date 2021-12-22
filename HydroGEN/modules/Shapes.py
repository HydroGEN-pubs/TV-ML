# Functions to calculate the ouput shapes from different layer types

def PFslice2D(array_in, slice_axis, lpick):
    # Function to take 2D slices of 3D model files
    # slic_axis - axis to slice along 1=x, 2=3, 3=z 
    # (i.e. slice wll be a plane parallel to the selected axis)
    #  lpick - the layer along that axis to select 
    # e.g. if slice_axis=3 and lpick=(nz-1)you will select the top layer of a model
    if slice_axis == 3:
        output = array_in[lpick, :, :]
    if slice_axis == 2:
        output=array_in[:, lpick, :]
    if slice_axis == 1:
        output = array_in[:, :, lpick]   
    
    return(output)


def conv2d_shape(input_shape=[1, 5, 5], cout=16, padding=1, kernel_size=3, dilation=1, stride=1):
    #Function to calculate the output shape of a conv2D layer given the shape of the
    #input array and the model settings.
    # Conv2D input shape: (Cin, Hin, Win)
    # Conv2D output shape: (Cout, Hout, Wout)
    # where:
    # N  = Batch Size
    # Cin = # of channels input and Cout=#channels output
    # Hout = [Hin + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    # Wout = [Win + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    Hin = input_shape[1]
    Win = input_shape[2]

    # Do it that way
    Hout = (Hin + 2 * padding - dilation *(kernel_size - 1) - 1) / stride + 1
    Wout = (Win + 2 * padding - dilation *(kernel_size - 1) - 1) / stride + 1

    output_shape = [cout, Hout, Wout]

    #Check if there are any non integer dimensions
    #mod = Hout % 1 + Wout % 1
    #if mod != 0:
    #    print('Note: Non integer dimensions resulting from Conv2D estimate')

    return(output_shape)

def convTranspose2d_shape(input_shape=[1, 5, 5], cout=16, padding=1, kernel_size=3, dilation=1, stride=1):
    #Function to calculate the output shape of a conv2D layer given the shape of the
    #input array and the model settings.
    # Conv2D input shape: (Cin, Hin, Win)
    # Conv2D output shape: (Cout, Hout, Wout)
    # where:
    # N  = Batch Size
    # Cin = # of channels input and Cout=#channels output
    # Hout = [Hin + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    # Wout = [Win + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    Hin = input_shape[1]
    Win = input_shape[2]

    # Do it that way
    
    Hout = (Hin -1)*stride - 2 * padding + dilation *(kernel_size - 1) + 1
    Wout = (Win -1)*stride - 2 * padding + dilation *(kernel_size - 1) + 1

    output_shape = [cout, Hout, Wout]

    #Check if there are any non integer dimensions
    #mod = Hout % 1 + Wout % 1
    #if mod != 0:
    #    print('Note: Non integer dimensions resulting from Conv2D estimate')

    return(output_shape)

def pool2d_shape(input_shape=[1, 5, 5], padding=0, kernel_size=2, dilation=1, stride=1):
    # Function to calculate the output shape of a pool2D layer given the shape of the
    # input array and the model settings.
    # This equaiton is the same as the CONV2D, but default kernel and padding are different and the cout=cin
    # Conv2D input shape: (Cin, Hin, Win)
    # Conv2D output shape: (Cout, Hout, Wout)
    # where:
    # N  = Batch Size
    # Cin = # of channels input and Cout=#channels output
    # Hout = [Hin + 2xpadding[0] - dilation[0]x(kernel_size[0] - 1) - 1]/stride[0] + 1
    # Wout = [Win + 2xpadding[1] - dilation[1]x(kernel_size[1] - 1) - 1]/stride[1] + 1
    Hin = input_shape[1]
    Win = input_shape[2]

    Hout = (Hin + 2 * padding - dilation *(kernel_size - 1) - 1) / stride + 1
    Wout = (Win + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    output_shape = [input_shape[0], Hout, Wout]

    #Check if there are any non integer dimensions
    #mod = Hout % 1 + Wout % 1
    #if mod != 0:
    #    print('Note: Non integer dimensions resulting from Pool2D estimate')

    return(output_shape)


def conv3d_shape(input_shape=[1, 5, 5, 5], cout=16, padding=1, kernel_size=3, dilation=1, stride=1):
    #Function to calculate the output shape of a conv3D layer given the shape of the
    #input array and the model settings.
    # Conv2D input shape: (Cin, Din, Hin, Win)
    # Conv2D output shape: (Cout, Hout, Wout)
    # where:
    # N  = Batch Size
    # Cin = # of channels input and Cout=#channels output
    # Dout = [Din + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    # Hout = [Hin + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    # Wout = [Win + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    Din = input_shape[1]
    Hin = input_shape[2]
    Win = input_shape[3]
 
    # Do it that way
    Dout = (Din + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    Hout = (Hin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    Wout = (Win + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    output_shape = [cout, Dout, Hout, Wout]

    #Check if there are any non integer dimensions
    mod = Hout % 1 + Wout % 1 + Dout % 1
    #if mod != 0:
    #    print('Note: Non integer dimensions resulting from Conv3D estimate')
    return(output_shape)


def pool3d_shape(input_shape=[1, 5, 5,5], padding=0, kernel_size=2, dilation=1, stride=1):
    # Function to calculate the output shape of a pool2D layer given the shape of the
    # input array and the model settings.
    # This equaiton is the same as the CONV2D, but default kernel and padding are different and the cout=cin
    # Conv2D input shape: (Cin, Hin, Win)
    # Conv2D output shape: (Cout, Hout, Wout)
    # where:
    # N  = Batch Size
    # Cin = # of channels input and Cout=#channels output
    # Dout = [Din + 2xpadding - (dilation x (kernel_size - 1)) - 1] / stride
    # Hout = [Hin + 2xpadding - (dilation x (kernel_size - 1)) - 1]/ stride + 1
    # Wout = [Win + 2xpadding - (dilation x (kernel_size - 1)) - 1]/ stride + 1
    Din = input_shape[1]
    Hin = input_shape[2]
    Win = input_shape[3]

    Dout = (Din + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    Hout = (Hin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    Wout = (Win + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    output_shape = [input_shape[0], Dout, Hout, Wout]

    #Check if there are any non integer dimensions
    mod = Hout % 1 + Wout % 1 + Dout % 1
    #if mod != 0:
    #    print('Note: Non integer dimensions resulting from pool3D estimate')

    return(output_shape)

