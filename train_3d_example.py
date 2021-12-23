# %%
### HydroGEN v0.001
# LE Condon & RM Maxwell
# Ml Model training script
# from:
# Maxwell, R.M.; Condon, L.E.; Melchior, P. 
# A Physics-Informed, Machine Learning Emulator of a 2D Surface Water Model: What Temporal Networks and Simulation-Based Inference Can Help Us Learn about Hydrologic Processes. 
# Water 2021, 13, 3633. https://doi.org/10.3390/w13243633

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random
import importlib
import yaml
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from functools import reduce

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# %%
# USER settings
# load command line arguments, the workflow path, yaml config file from the command line generated
# by the run_workflow.sh script. 
config_file = sys.argv[1]
this_file = sys.argv[0]
run_config = sys.argv[2]
print("Using config file:", config_file)

# %%
# USER settings
#Run selection settings
get_from_csv = True #if false run list will be generated from directory
n_run_train = -1  # Number of runs to train on: -1 means train on all runs found

run_pick = 'Random' #'Random' to pick runs randomly other option 'First' to select the first n runs

shuffle = True # Shuffle the rundefinition list before training to make sure to train in random order

# Spatial slicing - settings to get 2D slices
slice_axis = 3  # slice will be taken parallel to the chosen axis 1=x, 2=y, 3=z
lpick = 0  # choose a layer for slicing 3D data into 2D (not for z: 0=bottom)

# Ml Training settings
batch_size = 150 #batch size for training set =-1 to make the batch size equal to #runs in the ensemble
N = -1  #time steps per run to use set =-1 to set this by the run length.
nchannel = 4 #number of input channels that will be used for training

# ML learning settings
learning_rate = 10e-5
n_epoch = 2500
losssave=np.zeros((n_epoch,2))

# control printing
print_verbose = False

# %%
# setting up run directories from the yaml file
file = open(config_file)
settings = yaml.load(file, Loader=yaml.FullLoader)

#Get the paths to the wokflow directory and the modules and add  to the sys path
local_path = settings['local_path']
workflow_path = settings['workflow_path']
model_defs_path = os.path.join(workflow_path, 'HydroGEN/model_defs')
pf_path = os.path.join(local_path, 'parflow_runs')
output_path = os.path.join(local_path, 'ml_runs')
module_path = os.path.join(workflow_path, 'HydroGEN/modules')
results_path = os.path.join(workflow_path, 'ml_results')
sys.path.append(module_path)
sys.path.append(model_defs_path)

#Read the rest of the run configuration
TV_config = settings['TV_config']
train_ens = settings['train_ens']
ml_model = settings['ml_model']

# get the path for the training ensemble
ensemble_dir = (TV_config + "-" + train_ens)
ensemble_path = os.path.join(pf_path, ensemble_dir)

# Create the run name from the settings
run_name = (ml_model + "_" + run_config)
ML_model_out = run_name + '.pt'

# Create a directory for the run if it doesn't already exist
run_path = os.path.join(output_path, run_name)
if not os.path.exists(run_path):
    os.makedirs(run_path)

#copy the yaml file into this run directory
shutil.copy2(config_file, run_path)
#copy the run script file (literally this file) into this run directory
shutil.copy2(this_file, run_path)

# %%
# Import the models you want to use
mldef = importlib.import_module(ml_model)
print("Using model:")
print(mldef)

# import hehlper / data modules
import Shapes
from Shapes import PFslice2D
import utilities as utilities

# Import PF modules
from parflow import Run
from data_access import DataAccessor, MLArrayBuilder
from transform import float32_clamp_scaling

# %%
# Get the run information
# Get a full list of run definitions available
if get_from_csv== False:
    run_definitions_all = utilities.GetRunsFromFolder(ensemble_path)
    print("Assembling run list for", ensemble_path)
else:
    #note this function also returns the ensemble settings dataframe
    run_definitions_all, ensemble_settings  = utilities.GetRunsFromCSV(ensemble_path)
    print("Assembling run list from csv for", ensemble_path)

# If n_run_train is not set to -1 grab a subset of these runs
if n_run_train != -1:
    run_definitions, run_list = utilities.SubsetRuns(run_definitions_all, n_run_train, subset_type = run_pick)
else:
    run_definitions = run_definitions_all
    run_list = np.arange(0, len(run_definitions_all))

# If shuffle is true then shuffle the final run_definition list os training will be in random order
if shuffle == True:
    random.shuffle(run_definitions)

# Check the number of runs
n_run_all = len(run_definitions_all)
n_run = len(run_definitions)

# if batch size ==-1 set it to the number of runs that were found
if batch_size == -1:
    batch_size = n_run
else:
    batch_size = min(n_run, batch_size)

# calculate the number of batches that will be needed
n_batch = np.ceil(n_run/batch_size)
print("------------------------")
print("Ensemble Information: ")
print("Using", n_run, "runs out of", n_run_all, "in the ensemble")
if n_run_train != -1:
    print("Subset by", run_pick)
print("Training in", n_batch, 'batches of size', batch_size)

# %%
# Read in run[0] to get the static information
run_definition = run_definitions[0]
run = Run.from_definition(run_definition)
data = DataAccessor(run)

# Get the time information of the run
start_time = run.TimingInfo.StartTime
stop_time = run.TimingInfo.StopTime

#if N was set to -1 get it from the run length
if N == -1:
    N = len(data.times)

print("------------------------")
print("Using", N, "out of", len(data.times), "timesteps per run")
print()

#Get the dimensions of the grid this has to be fixed over all realizations
nx = run.ComputationalGrid.NX
ny = run.ComputationalGrid.NY
nz = run.ComputationalGrid.NZ
print("------------------------")
print(" Model Grid Information")
print("PF Grid Size (nx,ny,nz):", nx, ny, nz)
print()

#Based on the layer you are slicing figure out the size of the grid
if slice_axis == 1:
    n_dim1 = ny
    n_dim2 = nz
if slice_axis == 2:
    n_dim1 = nx
    n_dim2 = nz
if slice_axis == 3:
    n_dim1 = nx
    n_dim2 = ny

n_grid = n_dim1 * n_dim2
print("Slicing on axis", slice_axis, "Slecting Layer", lpick)
print("2D Grid Dimensions (n_dim1, n_dim2):", n_dim1, n_dim2)
print("2D Grid Size (n_grid):", n_grid)
print()

# %%
# Setup the ml_data structure
D_in = nchannel * n_grid
D_out = n_grid

print('-----------------------------')
print("  NN Model information")
print()

# Choose the ML and associated options, models are stored in ML_models.py
# and need to be imported up top if used
model = mldef.RMM_NN(grid_size=[N, n_dim2, n_dim1],
               channels=nchannel)
model.to(DEVICE)

model.verbose=False
model.use_dropout = True

print("-- Model Definition --")
print(model)

print("-- Model Parameters --")
utilities.count_parameters(model)

## options for different loss models and solvers
# loss function
#
loss_fn = torch.nn.SmoothL1Loss()

# optimizer and solver, Adam works the best, regular SGD seems to work the poorest
#

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup transforms to get variables scaled from 0-1; 
PRESS_TRANSF = float32_clamp_scaling(src_range=[0.0, 0.063], dst_range=[0, 1])
SLOPE_TRANSF = float32_clamp_scaling(src_range=[-0.2, 0.2], dst_range=[-1, 1])
MANN_TRANSF = float32_clamp_scaling(src_range=[5.00E-06, 2.08E-05], dst_range=[0, 1])
MAXP_TRANF = float32_clamp_scaling(src_range=[-0.02, 0.0], dst_range=[-1, 0])

print()
print("-- Model Settings --")
print(" Channels:",nchannel)
print(" Epochs:",n_epoch)
print(" Learning Rate:", learning_rate)
print()
print()

# %%
## loop over all the realization members
##
print("--------------------------------------------------------------")
print(" Training on", n_run," ensemble members")
print("Training with", n_batch, "batches with", batch_size, "runs per batch")
print()
#print("---------------------------------")
#print("  Training Ensemble Member:", count)
# Setup the input arrays
print("Reading Transient Variables and Seting Up Data")
print("Loading Ensemble Member Runs For Batch Processing")

## RMM, load all data and process inputs and labels up front
print()
print("____")
print("Processing all input data")

#setup the input and ouput arrays
input_temp = np.zeros((n_run, nchannel, N, n_dim2, n_dim1))
output_temp_temp = np.zeros((N, D_out))
output_temp = np.zeros((n_run, N*D_out))

count = 0
for rr in range(n_run):
    run_definition=run_definitions[count]
    count += 1
    progress = int(100 * count / len(run_definitions))

    # ----------------------------------------
    # # Setup run from ensemble
    # # ----------------------------------------
    # # load data
    run = Run.from_definition(run_definition)
    data = DataAccessor(run)

    # load rain data and duration from ParFlow keys
    maxp = run.Patch.z_upper.BCPressure.rain.Value
    time_application = int(run.Cycle.rainrec.rain.Length)
    if print_verbose == True:
        print()
        print("-----------------------------------------------")
        print(f'{progress}% loading {run_definition}')
        print()
        print("-- Precip Inputs --")
        print("Max Precip [m/h]:", maxp)
        print("Time of Application [h]:", (0.1*time_application))
        # ___
        # Setup the static variables you would like to use
        print("Reading Static Variables")
    else:
        print(f'{progress}% ', end='\r')


    # Max Precip is mapped to array
    Max_Precip = np.zeros((n_dim2, n_dim1))
    Max_Precip += maxp
    Zero_Precip = np.zeros((n_dim2, n_dim1))

    # ___
    # load all the transient variables for timesteps, ie the batch members and normalize
    for ii in range(0, N):
        # Static Variable are slopes
        input_temp[rr, 0, ii, :, :] = SLOPE_TRANSF(PFslice2D(data.slope_x, slice_axis, lpick))
        input_temp[rr, 1, ii, :, :] = SLOPE_TRANSF(PFslice2D(data.slope_y, slice_axis, lpick))
        input_temp[rr, 2, ii, :, :] = MANN_TRANSF(PFslice2D(data.mannings, slice_axis, lpick))
        # precip is time dependent
        if (ii<=time_application):
            input_temp[rr, 3, ii, :, :] = MAXP_TRANF(Max_Precip)
        else:
            input_temp[rr, 3, ii, :, :] = MAXP_TRANF(Zero_Precip)

        data.time = ii
        pressure = np.where(data.pressure<=0.0, 0.0, data.pressure)
        output_temp_temp[ii, :] = np.reshape(PRESS_TRANSF(
            PFslice2D(pressure, slice_axis, lpick)), n_grid)

    # now after the input and labels are assembled reshape arrays
    # over batch number
    output_temp[rr,:] = np.reshape(output_temp_temp, N*n_grid)

print()
print("Training:")
#LC - Flipping the loop here; the for epoch needs to go here  
for epoch in range(n_epoch): 
    optimizer.zero_grad()
    count = 0

    for bb in range(int(n_batch)):
        # Calculate how many runs in the batch
        # = n_batch unless its the last batch then its whatever is left
        if bb == (n_batch-1) and (n_run % batch_size)!=0:
            batch_size_temp = n_run % batch_size
        else:
            batch_size_temp = batch_size
            
        # this loop stays inside the batch loop, the input_temp and outout_temp are assembled 
        # over the entire set of runs
        # convert the inputs and outputs (labels) from the temp NumPy vectors to Torch format
        # convert them to Floats
        b_from = (bb*batch_size)
        b_to = batch_size_temp + (bb*batch_size)
        torch_input = torch.from_numpy(input_temp[b_from:b_to,:,:,:,:])
        torch_label = torch.from_numpy(output_temp[b_from:b_to,:])
        torch_input = torch_input.type(torch.FloatTensor).to(DEVICE)
        torch_label = torch_label.type(torch.FloatTensor).to(DEVICE)
        # ___
        # LC-- I think all of this goes here and its only the optimizer.zero grad that needs to go outside the batch loop
        batch_prediction = model(torch_input)
        loss = loss_fn(batch_prediction, torch_label)
        print("Epoch: %3d, batch: %3d, loss: %5.3e" % (epoch, bb, loss), end='\r')
        losssave[epoch,0] = epoch
        losssave[epoch,1] = loss
        loss.backward()
        optimizer.step()

print()
#save loss
lossoutfile = os.path.join(results_path,(ml_model+'.csv'))
print(lossoutfile)
np.savetxt(lossoutfile, losssave, delimiter=',')
print("____")
print("Training Complete")

# %%
# Save our progress
#trainer.save(model_path)
print("-----------------------------------")
print(" Saving the model")
model_path = os.path.join(run_path, ML_model_out)
print(model_path)
torch.save(model, model_path)
