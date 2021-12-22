#utility functions for handling ParFlow runs
import os
import pandas as pd
import numpy as np
import random
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def GetDrainageArea(n_dim2, n_dim1):
    '''
    Makes a drainage area map for a tilted v given ny and nx

    Assumes that the outlet of the domain is at y=0 and x=nx/2

        Parameters:
            n_dim2 (int) : shape of the domain in y
            n_dim2 (int) : shape of the domain in x

        Returns:
            area (nparray): n_dim2 by n_dim1 array of drainag areas
    '''
    
    area = np.zeros((n_dim2, n_dim1))
    hill_left = np.arange(0, int((n_dim1-1)/2))
    hill_right = np.flip(hill_left)

    area[:, 0:int((n_dim1-1)/2)] = hill_left
    area[:, int(np.ceil((n_dim1)/2)):] = hill_right
    area[:, int(np.floor((n_dim1)/2))] = np.arange(stop=(n_dim1-1),
                                                start=(n_dim2*n_dim1), step=-n_dim1)-1
    return(area)


def GetRunsFromFolder(ensemble_dir):
    '''
    Returns a list of pfidb paths give a run directory

        Parameters:
            enseble_dir (Path) : Path to a directory to search

        Returns:
            run_definitions (list): List of pfidb paths
    ''' 

    run_definitions = []
    for root, dirs, files in os.walk(ensemble_dir):
        for file in files:
            if file.endswith(".pfidb"):
                file_path = os.path.join(root, file)
                run_definitions.append(file_path)
                nrun_total = len(run_definitions)

    return(run_definitions)


def GetRunsFromCSV(ensemble_dir):
    '''
    Returns a list of pfidb paths give an enseble csv list

        Parameters:
            enseble_dir (Path) : Path to a directory to search

        Returns:
            run_definitions (list): List of pfidb paths
            ensemble_setting (DataFrame): Pandas data frame of ensemble settings
    '''
    ensemble_settings = pd.read_csv(
        os.path.join(ensemble_dir, 'ensemble_settings.csv'))
    n_run = len(ensemble_settings)

    #convert the folder numbers to the folder name strings with zero padding
    padding = len(str(len(ensemble_settings)))
    dir_list = ensemble_settings["path"].astype(str).str.zfill(padding)

    #create the run_deffinitions
    run_definitions = []
    for ii in range(n_run):
        pfidb = ensemble_settings.run_name[ii]+'.pfidb'
        file_path = os.path.join(
            ensemble_dir, dir_list[ii], pfidb)
        run_definitions.append(file_path)

    return(run_definitions, ensemble_settings)

def SubsetRuns(run_definitions,  n_run, subset_type='Random'):
    '''
    Returns a subset of the run definition list

        Parameters:
            run_definitions (list) : List of run definition file paths
            n_run : The number of runs you would like to select
            subset_type: the way you would like to subset - options are 'Random', 'First',  'Even' 

        Returns:
            run_definitions (list): Subset list of pfidb paths that will be n_run long
            run_list (list): An index list of the runs that were selected
    '''
    run_definitions_all=run_definitions
    n_run_all = len(run_definitions_all)

    if 0 < n_run <= n_run_all:
        if subset_type == 'Random':
            run_list = random.sample(range(n_run_all), n_run)
            run_definitions = [run_definitions_all[i] for i in run_list]
        elif subset_type == 'First':
            run_list = np.arange(0,n_run)
            run_definitions = run_definitions_all[0:n_run]
        elif subset_type == 'Even':
            run_list = np.arange(0, n_run_all, 2, dtype=int)
            run_definitions = [run_definitions_all[i] for i in run_list]
        else:
            print("Must select valid subset type: choices are Random, First or Even")
    else:
        print('Not subsetting: n_run is less than 0 or greater than the number of runs provided')
    
    return(run_definitions, run_list)
