# %%
### HydroGEN v0.001
# LE Condon & RM Maxwell
# Ml Model eval script
# from:
# Maxwell, R.M.; Condon, L.E.; Melchior, P. 
# A Physics-Informed, Machine Learning Emulator of a 2D Surface Water Model: What Temporal Networks and Simulation-Based Inference Can Help Us Learn about Hydrologic Processes. 
# Water 2021, 13, 3633. https://doi.org/10.3390/w13243633# %%

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import random
import importlib
import yaml
import shutil
from scipy import stats
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from functools import reduce

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
# load command line arguments, the workflow path, yaml config file from the command line generated
# by the run_workflow.sh script.  
config_file = sys.argv[1]  
this_file = sys.argv[0]
run_config = sys.argv[2]
print("Using config file:", config_file)

#  %%
print("---------------------------------------------")
print("Loading metadata, setting up model grid info")
print("---------------------------------------------")
print()

# USER settings
get_from_csv = True #if false run list will be generated from directory
n_run_eval = -1  # Number of runs to train on: -1 means train on all runs found
run_pick = 'Random' #'Random' to pick runs randomly other option 'First' to select the first n runs
shuffle = True # Shuffle the run definition list 

N = -1  #time steps per run to use set =-1 to set this by the run length. 
nchannel = 5 #number of input channels that will be used for training

# Choose the axis you are are taking a slice along and the layer to be used
# this will take you from 3D inputs to 2D  slices
slice_axis = 3  #1=x, 2=y, 3=z
lpick = 0 # choose a layer for slicing 3D data into 2D

plot_runs = True 
plot_runs = False

plot_summary=True

print_verbose = False

# Max / Min Pressure tracker
PFmaxP=0.0
PFminP=0.0
# %%
# %%
# setting up run directories from the yaml file
file = open(config_file)
settings = yaml.load(file, Loader=yaml.FullLoader)

#Get the paths to the wokflow directory and the modules and add  to the sys path
local_path = settings['local_path']
workflow_path = settings['workflow_path']
model_defs_path = os.path.join(workflow_path, 'HydroGEN/model_defs')
pf_path = os.path.join(local_path, 'parflow_runs')
module_path = os.path.join(workflow_path, 'HydroGEN/modules')
output_path = os.path.join(local_path, 'ml_runs')
results_path = os.path.join(workflow_path, 'ml_results')
sys.path.append(module_path)
sys.path.append(model_defs_path)

#Read the rest of the run configuraiton
TV_config = settings['TV_config']
train_ens = settings['train_ens']
eval_ens_list = settings['eval_ens']
ml_model = settings['ml_model']

# Get the run name, model file and run directory from the settings
run_name = (ml_model + "_" + run_config)
run_path = os.path.join(output_path, run_name)
model_path = os.path.join(run_path, (run_name + '.pt'))

shutil.copyfile(config_file, os.path.join(run_path, ("eval_"+config_file)))

shutil.copy2(this_file, run_path)
#  %%

import Shapes
from Shapes import PFslice2D
import utilities as utilities
import eval_tools as evtools

from parflow import Run
from data_access import DataAccessor, MLArrayBuilder
from transform import float32_clamp_scaling
# %%
#loop over the evaluation ensembles to be evaluated
for ens in range(len(eval_ens_list)):
    eval_ens = eval_ens_list[ens]
    ensemble_dir = (TV_config + "-" + eval_ens)
    ensemble_path = os.path.join(pf_path, ensemble_dir)
    eval_name = run_name + "_Eval." + eval_ens
    print("Evaluating Ensemble:", eval_ens)

    # Get a full list of run definitions available
    if get_from_csv== False:
        run_definitions_all = utilities.GetRunsFromFolder(ensemble_path)
        print("Assembling run list for", ensemble_path)
    else:
        #note this function also returns the ensemble settings dataframe
        run_definitions_all, ensemble_settings  = utilities.GetRunsFromCSV(ensemble_path)
        print("Assembling run list from csv for", ensemble_path)

    # If n_run_eval is not set to -1 grab a subset of these runs
    if n_run_eval != -1:
        run_definitions, run_list = utilities.SubsetRuns(run_definitions_all, n_run_eval, subset_type = run_pick)
    else:
        run_definitions = run_definitions_all
        run_list = np.arange(0, len(run_definitions_all))

    # Check the number of runs
    n_run_all = len(run_definitions_all)
    n_run = len(run_definitions)

    # If shuffle is true then shuffle the final run_definition list os training will be in random order
    if shuffle == True:
        ilist= np.arange(0,n_run)
        random.shuffle(ilist)
        run_definitions  = [run_definitions[i] for i in ilist]
        run_list = [run_list[i] for i in ilist]

    # %%
    # Setup transforms to get variables scaled from 0-1

    PRESS_TRANSFL10 = float32_clamp_scaling(src_range=[0.0, .063], dst_range=[0, 1])
    PRESS_TRANSF = float32_clamp_scaling(src_range=[0.0, .07], dst_range=[0, 1])
    PERM_TRANSF = float32_clamp_scaling(src_range=[0, 1], dst_range=[0, 1])
    SATUR_TRANSF = float32_clamp_scaling(src_range=[0, 1], dst_range=[0,1])
    SLOPE_TRANSF = float32_clamp_scaling(src_range=[-0.2, 0.2], dst_range=[-1, 1])
    MAXP_TRANF = float32_clamp_scaling(src_range=[0.0, 0.02], dst_range=[0, 1])
    MANN_TRANSF = float32_clamp_scaling(src_range=[5.00E-06, 2.08E-05], dst_range=[0, 1])
    INV_PRESS_TRANSFL10 = float32_clamp_scaling(src_range=[0, 1], dst_range=[0.0, 0.063])
    # %%
    # Read in run[0] to get the static information
    run_definition = run_definitions[0]

    # load the PF metadata and put it the run data structure
    run = Run.from_definition(run_definition)

    #Get the dimensions of the grid this has to be fixed over all realizations
    nx = run.ComputationalGrid.NX
    ny = run.ComputationalGrid.NY
    nz = run.ComputationalGrid.NZ
    print("------------------------")
    print(" Model Grid Information")
    print("PF Grid Size (nx,ny,nz):", nx, ny, nz)
    print()

    #based on the layer you are slicing figure out the size of the grid
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

    #if N was set to -1 get it from the run length
    data = DataAccessor(run)
    if N == -1:
        N = len(data.times)

    # %%
    # ML load and setup

    print('-----------------------------')
    print("  NN Model information")
    print()

    print()
    print("-- Model Definition --")
    # %%
    # Load model

    print("------------------------------------")
    print(" Loading the model from:")
    print(model_path)
    model=torch.load(model_path, map_location=DEVICE)
    model.eval()
    print(model)

    # set up hydrograph array
    # column 1 time, then PF=0 or ML=1, then n_run_eval
    hydrograph = np.zeros((N,2,n_run))
    outlet_metrics = np.zeros((n_run, 8))
    outlet_loc = [0, int(np.floor(n_dim1/2))] # location of the outlet point where metrics will be recorded

    eval_metrics = np.zeros((n_run, 22))
    #column names for eval metrics
    eval_names = ['PF_peak_outlet', 'ML_peak_outlet',
                'PF_peaktime_outlet', 'ML_peaktime_outlet',
                'PF_sum_outlet', 'ML_sum_outlet',
                'rmse_outlet', 'total_diff_outlet',
                'PF_peak_mean', 'ML_peak_mean',
                'PF_peaktime_mean', 'ML_peaktime_mean',
                'PF_sum_mean', 'ML_sum_mean',
                'rmse_mean', 'total_diff_mean','rmse_real',
                'rmse_hydrograph', 'Pearson_hydrograph',
                'Spearman_hydrograph','PF_sum_hydrograph',
                'ML_sum_hydrograph']

    count = 0
    for run_definition in run_definitions:
        count += 1
        progress = int(100 * count / len(run_definitions))

        # ----------------------------------------
        # # Setup run from ensemble
        # # ----------------------------------------
        # # load data
        run = Run.from_definition(run_definition)
        data = DataAccessor(run)
    
        # Get the time information of the run
        start_time = run.TimingInfo.StartTime
        stop_time = run.TimingInfo.StopTime

        #if N was set to -1 get it from the run length
        if N == -1:
            N = len(data.times)

        #--------
        # Setup the static variables you would like to use
        # -------
        maxp = np.abs(run.Patch.z_upper.BCPressure.rain.Value)
        time_application = int(run.Cycle.rainrec.rain.Length)

        Max_Precip = np.zeros((n_dim2, n_dim1))
        Max_Precip += maxp
        Zero_Precip = np.zeros((n_dim2, n_dim1))

        # Setup the input arrays, the model input is just one timestep (the first one)
        # and the model comparison is now all the timesteps read from that PF run
        # Note there is no label vector now
        input_temp = np.zeros((1, nchannel, n_dim2, n_dim1))
        #Pressure at time t=0
        data.time = data.times[0]

        # create a model output vector
        model_output = np.zeros((N, n_dim2, n_dim1))
        # create a PF output vector
        PF_output = np.zeros((N, n_dim2, n_dim1))

        #Pressure at time t=0, input to ML and PF out array
        data.time = 0
        model_output[0,:,:] = PFslice2D(data.pressure, slice_axis, lpick)
        PF_output[0,:,:] = PFslice2D(data.pressure, slice_axis, lpick)
        hydrograph[0,1,count-1] = evtools.calc_hydrograph(model_output[0,0,12],data.slope_y[0,0,12],data.mannings[0,0,12])
        hydrograph[0, 0, count-1] = evtools.calc_hydrograph(
                PF_output[0, 0, 12], data.slope_y[0, 0, 12], data.mannings[0, 0, 12])
        if print_verbose == True:
            print()
            print("-----------------------------------------------")
            print(f'{progress}% Comparing to {run_definition}')
            print()
            print("PF Run start = ", start_time, "Run Stop = ", stop_time)
            print("Number of Timesteps =", N)
            print()
            print("Reading Transient Variables and Seting Up Run Data")
            print("Reading Static Variables")
            print()
            print("-- Precip Inputs --")
            print(maxp)     
            # Make Predictions
            print("-----------------------------------")
            print(" Make Predictions")
            print
        # freeze the model
        model.use_dropout = False
        for parameter in model.parameters():
            parameter.requires_grad = False
        # 
        # load all the transient variables for timesteps, and normalize the model inputs only
        for ii in range(0, N-1):
            print("Batch Progress: %3.0f%%  model progress: %3.0f%% " % (progress, 100*(ii/N)), end='\r')
            data.time = ii
            input_temp[0, 0, :, :] = SLOPE_TRANSF(PFslice2D(data.slope_x, slice_axis, lpick))
            input_temp[0, 1, :, :] = SLOPE_TRANSF(PFslice2D(data.slope_y, slice_axis, lpick))
            input_temp[0, 2, :, :] = MANN_TRANSF(PFslice2D(data.mannings, slice_axis, lpick))
            if (ii<=time_application):
                input_temp[0, 3, :, :] = MAXP_TRANF(Max_Precip)
            else:
                input_temp[0, 3, :, :] = MAXP_TRANF(Zero_Precip)
            data.time = ii
            input_temp[0,4,:,:] = PRESS_TRANSFL10(model_output[ii,:,:])

            
            # copy into PF model output array 
            # convert the inputs and outputs (labels) from the temp NumPy vectors to Torch format
            predict_input = torch.from_numpy(input_temp)
            # convert to Floats
            predict_input = predict_input.type(torch.FloatTensor).to(DEVICE)
            prediction = model(predict_input)
            # copy into ML model output array
            #
            model_output[ii+1,:,:] = INV_PRESS_TRANSFL10(
                np.reshape(prediction.data.cpu().numpy(), (n_dim2, n_dim1)))
            hydrograph[ii+1,1,count-1] = evtools.calc_hydrograph(model_output[ii+1,0,12],data.slope_y[0,0,12],data.mannings[0,0,12])
            data.time = ii+1
            PF_output[ii+1,:, :] = PFslice2D(np.where(data.pressure <= 0.0, 0.0, data.pressure), slice_axis, lpick)
            PFmaxP=max(np.max(PF_output[ii,:,:]), PFmaxP)
            PFminP=min(np.min(PF_output[ii,:,:]), PFminP)
            hydrograph[ii+1, 0, count-1] = evtools.calc_hydrograph(
                PF_output[ii+1, 0, 12], data.slope_y[0, 0, 12], data.mannings[0, 0, 12])
    
        #_______________
        # Plot simulated and observed values and print overall RMSE
        #_______________
        if plot_runs == True:
            fig, axs = plt.subplots(6, 3, constrained_layout=True)
            axs[0,0].imshow(model_output[0,:,:], cmap='hot')
            axs[0,0].set_title('ML output')
            axs[0,1].imshow(PF_output[0,:,:], cmap='hot') 
            axs[0,1].set_title('PF output time: %4.1f' % 0)
            axs[0,2].imshow(PF_output[0,:,:]-model_output[0,:,:], cmap='viridis_r')
            axs[0,2].set_title('PF-ML difference')
            for ii in range(1,6):
                jj = np.int(ii*(N/5))-1
                print(" RMSE at time %6.0f : %6.3e" % (data.times[jj],
                    evtools.rmse(PF_output[jj,:,:],model_output[jj,:,:])))
                axs[ii,0].imshow(model_output[jj,:,:], cmap='hot')
                axs[ii,1].set_title('time: %4.1f' % (jj*.05))
                axs[ii,1].imshow(PF_output[jj,:,:], cmap='hot') 
                axs[ii,2].imshow(PF_output[jj,:,:]-model_output[jj,:,:], cmap='viridis_r')
            for axs in fig.get_axes():
                axs.label_outer()
            plt.show()

        #_______________
        # Calculate Hydrograph metrics at every grid cell and plot
        #_______________
        #Hydrograph peak value at every grid cell
        PF_peak = evtools.hydgrph_3D_peakval(PF_output)
        ML_peak = evtools.hydgrph_3D_peakval(model_output)

        #Time to peak at every grid cell
        PF_tpeak = evtools.hydgrph_3D_peaktime(PF_output, dt=run.TimeStep.Value)
        ML_tpeak = evtools.hydgrph_3D_peaktime(model_output, dt=run.TimeStep.Value)

        #Hydrograph total at every grid cell
        PF_sum = evtools.hydgrph_3D_total(PF_output, dt=run.TimeStep.Value)
        ML_sum = evtools.hydgrph_3D_total(model_output, dt=run.TimeStep.Value)

        #RMSE between hydrographs at every grid cell
        rmse_map = evtools.hydgrph_3D_rmse(model_output, PF_output)

        #RMSE between hydrographs at every grid cell
        dif_map = evtools.hydgrph_3D_rmse(model_output, PF_output)

        #RMSE between hydrographs at every grid cell
        dif_map = evtools.hydgrph_3D_cumdiff(model_output, PF_output)

        # Make a multi panel plot showing this
        if plot_runs == True:
            fig, ax = plt.subplots(4, 2, sharex='col', sharey='row',
                                constrained_layout=False)
            ax[0, 0].imshow(PF_peak, cmap='viridis_r')
            ax[0, 1].imshow(ML_peak, cmap='viridis_r')
            ax[1, 0].imshow(PF_tpeak, cmap='viridis_r')
            ax[1, 1].imshow(ML_tpeak, cmap='viridis_r')
            ax[2, 0].imshow(PF_sum, cmap='viridis_r')
            ax[2, 1].imshow(ML_sum, cmap='viridis_r')
            ax[3, 0].imshow(rmse_map, cmap='viridis_r')
            ax[3, 1].imshow(dif_map, cmap='viridis_r')
            plt.show()
    # EM 0 'PF_peak_outlet'
    # EM 1 'ML_peak_outlet'
    # EM 2 'PF_peaktime_outlet'
    # EM 3 'ML_peaktime_outlet'
    # EM 4 'PF_sum_outlet'
    # EM 5 'ML_sum_outlet'
    # EM 6 'rmse_outlet'
    # EM 7 'total_diff_outlet'
    # EM 8 'PF_peak_mean'
    # EM 9 'ML_peak_mean'
    # EM 10'PF_peaktime_mean'
    # EM 11'ML_peaktime_mean'
    # EM 12'PF_sum_mean'
    # EM 13'ML_sum_mean'
    # EM 14'rmse_mean'
    # EM 15'total_diff_mean'
    # EM 16'rmse_real' (RMSE between ML and PF over the entire realization)
    # EM 17'rmse_hydrograph' (RMSE between ML and PF over the outflow hydrograph)
    # EM 18'Pearson_hydrograph' (pearson correlation coeff ML PF over outflow hydrograph)
    # EM 19'Spearman_hydrograph' (spearman coeff ML PF over outflow hydrograph)
    # EM 20'PF_sum_hydrograph' (sum of PF outflow hydrograph)
    # EM 21'ML_sum_hydrograph' (sum of ML outflow hydrograph)

        # Record the metrics for the outlet
        eval_metrics[(count-1), 0] = np.max(hydrograph[:,0,(count-1)])
        eval_metrics[(count-1), 1] = np.max(hydrograph[:,1,(count-1)])
        eval_metrics[(count-1), 2] = evtools.hydgrph_3D_peaktime(hydrograph[:,0,(count-1)], dt=run.TimeStep.Value)
        eval_metrics[(count-1), 3] = evtools.hydgrph_3D_peaktime(hydrograph[:,1,(count-1)], dt=run.TimeStep.Value)
        eval_metrics[(count-1), 4] = PF_sum[outlet_loc[0], outlet_loc[1]]
        eval_metrics[(count-1), 5] = ML_sum[outlet_loc[0], outlet_loc[1]]
        eval_metrics[(count-1), 6] = rmse_map[outlet_loc[0], outlet_loc[1]]
        eval_metrics[(count-1), 7] = dif_map[outlet_loc[0], outlet_loc[1]]

        # Record average metrics 
        eval_metrics[(count-1), 8] = np.mean(PF_peak)
        eval_metrics[(count-1), 9] = np.mean(ML_peak)
        eval_metrics[(count-1), 10] = np.mean(PF_tpeak)
        eval_metrics[(count-1), 11] = np.mean(ML_tpeak)
        eval_metrics[(count-1), 12] = np.mean(PF_sum)
        eval_metrics[(count-1), 13] = np.mean(ML_sum)
        eval_metrics[(count-1), 14] = np.mean(rmse_map)
        eval_metrics[(count-1), 15] = np.mean(dif_map)

        # Record RMSE over the entire domain and hydrograph stats
        eval_metrics[(count-1), 16] = evtools.rmse(PF_output[:,:,:],model_output[:,:,:])
        eval_metrics[(count-1), 17] = evtools.rmse(hydrograph[:,1,(count-1)],hydrograph[:,0,(count-1)])
        eval_metrics[(count-1), 18] = (np.corrcoef(hydrograph[:,1,(count-1)],hydrograph[:,0,(count-1)])[0,1])**2
        eval_metrics[(count-1), 19] = stats.spearmanr(hydrograph[:,1,(count-1)],hydrograph[:,0,(count-1)])[0]
        eval_metrics[(count-1), 20] = np.sum(hydrograph[:,0,(count-1)])
        eval_metrics[(count-1), 21] = np.sum(hydrograph[:,1,(count-1)])
    print()
    # %%
    # Write out csv's for the metrics and the hydrographs
    # Make a dataframe of the settings
    metrics_df = pd.DataFrame(eval_metrics, columns=eval_names)
    metrics_df.index = run_list
    csvfile = os.path.join(run_path, (eval_name + '_ML_metrics.csv'))
    metrics_df.to_csv(csvfile, index=True)

    pf_df = pd.DataFrame(hydrograph[:, 0, :], columns=run_list)
    csvfile = os.path.join(run_path, (eval_name + '_PF_Hydographs.csv'))
    pf_df.to_csv(csvfile, index=False)

    ml_df = pd.DataFrame(hydrograph[:, 1, :], columns=run_list)
    csvfile = os.path.join(run_path, (eval_name + '_ML_Hydrographs.csv'))
    ml_df.to_csv(csvfile, index=False)

    plotpdffile = os.path.join(
        results_path, (eval_name + '_summary_plots.pdf'))
    pp = PdfPages(plotpdffile)

    markdownoutfile = os.path.join(
        results_path, (eval_name + '_summary_stats.md'))
    markdownFile = open(markdownoutfile, 'w')

    # %%
    # Plot hydrograph at outlet
    #plot hydrograph2
    if plot_summary == True:
        x_axis=[x * 0.05 for x in range(0, N)]

        one = hydrograph[:,0,:].mean(axis=1)
        minone = hydrograph[:,0,:].min(axis=1)
        maxone = hydrograph[:,0,:].max(axis=1)
        two = hydrograph[:,1,:].mean(axis=1)
        mintwo = hydrograph[:,1,:].min(axis=1)
        maxtwo = hydrograph[:,1,:].max(axis=1)
        plt.plot(x_axis,one,label=('ParFlow'),color='blue', linewidth=3)
        plt.plot(x_axis,two,label=('PF-ML'),color='red', linewidth=3)
        plt.plot(x_axis,minone,color='blue', linewidth=1, dashes=[3,1])
        plt.plot(x_axis,mintwo,color='red', linewidth=1,dashes=[3,1])
        plt.plot(x_axis,maxone,color='blue', linewidth=1, dashes=[3,1])
        plt.plot(x_axis,maxtwo,color='red', linewidth=1,dashes=[3,1])
        plt.fill_between(x_axis, minone, maxone,color='blue', alpha=0.1)
        plt.fill_between(x_axis, mintwo, maxtwo,color='red', alpha=0.1)
        plt.ylabel("Outflow [m^2/h]")
        plt.xlabel("Time [h]")
        plt.title("Hydrograph Ensemble Comparison")
        plt.legend()
        plt.savefig(pp, format='pdf')
        plt.close()

    #plot the PF vs ML peak value

    # %%
    #plot outlet metrics
    if plot_summary == True:
        fig,ax=plt.subplots(3,1)
        plt.subplots_adjust(hspace=1)

        fig.suptitle("Hydrograph Metrics at the outlet")
        ax[0].plot(eval_metrics[:, 0], eval_metrics[:, 1],
                color='blue', marker='o', ls="")
        ax[0].set_title("Peak Value")
        plotmin = np.min(eval_metrics[:, 0:2])
        plotmax = np.max(eval_metrics[:, 0:2])
        ax[0].plot((plotmin, plotmax), (plotmin, plotmax), color='grey', ls=':')
        ax[0].set_xlim(plotmin*0.9, plotmax*1.1)
        ax[0].set_ylim(plotmin*0.9, plotmax*1.1)
        ax[0].set_xlabel("ParFlow")
        ax[0].set_ylabel("ML")

        ax[1].plot(eval_metrics[:, 2], eval_metrics[:, 3],
                   color='blue', marker='o', ls="")
        ax[1].set_title("Peak Time")
        plotmin = np.min(eval_metrics[:, 2:4])
        plotmax = np.max(eval_metrics[:, 2:4])
        ax[1].plot((plotmin, plotmax), (plotmin,plotmax), color='grey', ls=':')
        ax[1].set_xlim(plotmin*0.9, plotmax*1.1)
        ax[1].set_ylim(plotmin*0.9, plotmax*1.1)
        ax[1].set_xlabel("ParFlow")
        ax[1].set_ylabel("ML")

        ax[2].plot(eval_metrics[:, 20], eval_metrics[:, 21],
                   color='blue', marker='o', ls="")
        ax[2].set_title("Total Outflow")
        plotmin = np.min(eval_metrics[:, 20:21])
        plotmax = np.max(eval_metrics[:, 20:21])
        ax[2].plot((plotmin, plotmax), (plotmin, plotmax), color='grey', ls=':')
        ax[2].set_xlim(plotmin, plotmax)
        ax[2].set_ylim(plotmin, plotmax)
        ax[2].set_xlabel("ParFlow")
        ax[2].set_ylabel("ML")
        plt.savefig(pp, format='pdf')
        plt.close()

    if plot_summary == True:
        fig,ax=plt.subplots(4,1)
        plt.subplots_adjust(hspace=1)
        num_bins = 150
        fig.suptitle("Aggregate Stats")
        ax[0].hist(eval_metrics[:,16], num_bins, density=1)
        ax[1].hist(eval_metrics[:,17], num_bins, density=1)
        ax[2].hist(eval_metrics[:,18], num_bins, density=1)
        ax[3].hist(eval_metrics[:,19], num_bins, density=1)
        ax[0].set_xlabel("RMSE of Pressure")
        ax[1].set_xlabel("RMSE of Hydrograph")
        ax[2].set_xlabel("Pearson of Hydrograph")
        ax[3].set_xlabel("Spearman of Hydrograph")

        #plt.show()
        plt.savefig(pp, format='pdf')
        plt.close()


    # %%
    ## print out summary output stats to screen and text file
    print()
    print("================================")
    print("     Summary Statistics")
    print(eval_name)
    ## print to markdown file
    print("# Summary Statistics",file = markdownFile)
    print("================================",file = markdownFile)
    print(file = markdownFile)
    print("# Model File:",file = markdownFile)
    print(eval_name, file=markdownFile)
    print(file = markdownFile)
    print("# Model Details:",file = markdownFile)
    print(model,file = markdownFile)
    from prettytable import PrettyTable

    table = PrettyTable()

    table.field_names = ['Quantity','Mean','Variance']
    table.add_row(['Pressure RMSE',"%6.3e" % np.mean(eval_metrics[:,16]), "%6.3e" % np.var(eval_metrics[:,16])])
    table.add_row(['Hydrograph RMSE', "%6.3e" %np.mean(eval_metrics[:,17]),"%6.3e" % np.var(eval_metrics[:,17])])
    table.add_row(['Hydrograph Pearson', "%4.3f"%np.mean(eval_metrics[:,18]),"%6.3e" %np.var(eval_metrics[:,18])])
    table.add_row(['Hydrograph Spearman',"%4.3f"%np.mean(eval_metrics[:,19]),"%6.3e" %np.var(eval_metrics[:,19])])
    print()
    print(" Mean, Var over all realizations")
    from prettytable import MARKDOWN

    print(table)
    table.set_style(MARKDOWN)
    print(file = markdownFile)
    print("# Summary Tables",file = markdownFile)
    print(file=markdownFile)
    print("# Mean, Var over all realizations", file=markdownFile)
    print(file=markdownFile)
    print(table, file = markdownFile)

    table = PrettyTable()
    table.field_names = ['Quantity','RMSE','Pearson', 'Spearman']
    temp_rmse = evtools.rmse(eval_metrics[:, 0], eval_metrics[:, 1])
    temp_pear = (np.corrcoef(eval_metrics[:, 0], eval_metrics[:, 1])[0,1])**2
    temp_spear = stats.spearmanr(eval_metrics[:, 0], eval_metrics[:, 1])[0]
    table.add_row(['Hydrograph Peak Outflow',"%6.3e" %temp_rmse, "%4.3f"%temp_pear, "%4.3f"%temp_spear])
    temp_rmse = evtools.rmse(eval_metrics[:, 2], eval_metrics[:, 3])
    temp_pear = (np.corrcoef(eval_metrics[:, 2], eval_metrics[:, 3])[0,1])**2
    temp_spear = stats.spearmanr(eval_metrics[:, 2], eval_metrics[:, 3])[0]
    table.add_row(['Hydrograph Peak Time',"%6.3e" %temp_rmse, "%4.3f"%temp_pear, "%4.3f"%temp_spear])
    temp_rmse = evtools.rmse(eval_metrics[:, 20], eval_metrics[:, 21])
    temp_pear = (np.corrcoef(eval_metrics[:, 20], eval_metrics[:, 21])[0,1])**2
    temp_spear = stats.spearmanr(eval_metrics[:, 20], eval_metrics[:, 21])[0]
    table.add_row(['Total Outflow',"%6.3e" %temp_rmse, "%4.3f"%temp_pear, "%4.3f"%temp_spear])
    print()
    print(" RMSE and Correlation between PF and ML")
    print(table)
    table.set_style(MARKDOWN)
    print(file=markdownFile)
    print("# RMSE and Correlation between PF and ML",file=markdownFile )
    print(file=markdownFile)
    print(table, file = markdownFile)
    pp.close()
    markdownFile.close()


    # %%
    print()
    print("Max PF Pressure:",PFmaxP)
    print("Min PF Pressure:",PFminP)
    print()
    print()
    print("===========================")
    print()