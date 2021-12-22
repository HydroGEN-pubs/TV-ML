#functions evaluating PF tilted V runs
import numpy as np

def calc_hydrograph(pressure, slope, mannings):
    '''
        Calculate streamflow for a single cell given pressure slope and manning
    '''
    # Function to calculate simple hdyrograph
    pressure = np.where(pressure < 0.0, 0., pressure)
    output = pow(pressure, (5/3))*pow(slope, (1/2))/(mannings)
    return(output)

def rmse(predictions, targets):
    '''
        Calculate rood mean square error between prediction and targets
    '''
    return np.sqrt(((predictions - targets) ** 2).mean())

#RMSE between hydrographs at a point
def hydgrph_3D_rmse(simulated_output, target_output):
    '''
        Calculate the rmse between hydrographs  

        Assumes that data are generated in 3D arrays where the first dimension is time such that
        sumulated_output[:,j,i] would be the timeseries hydrograph at a single point

        Returns a spatial map of values with the rmse for each grid cell. 
        
        Parameters:
            simulated_output (nparray) : (nt, n_dim2, n_dim2) array of outputs from the ML simulation
            target_output (nparray) : (nt, n_dim2, n_dim2) array of target outputs
        Returns:
            dif_map (nparray): n_dim2 by n_dim1 array of the total hydrograph rmse values
    '''
    n_dim2 = simulated_output.shape[1]
    n_dim1 = target_output.shape[2]
    
    rmse_map = np.zeros((n_dim2, n_dim1))

    for jj in range(n_dim2):
        for ii in range(n_dim1):
            dif = simulated_output[:, jj, ii] - target_output[:, jj, ii]
            dif2 = dif**2
            rmse_map[jj, ii] = np.sqrt(dif2.mean())

    return(rmse_map)

#Time of hydograph peak 
def hydgrph_3D_peaktime(output_array, dt=1):
    '''
        Calculate the time to peak for hydrographs 

        Assumes that data are generated in 3D arrays where the first dimension is time such that
        sumulated_output[:,j,i] would be the timeseries hydrograph at a single point

        Returns a spatial map of values with the rmse for each grid cell. 
        
        Parameters:
            output_array (nparray) : (nt, n_dim2, n_dim2) array of values
            dt (nparray) : the timestep for the simulation, default = 1
        Returns:
            tprsk (nparray): n_dim2 by n_dim1 array of the time to peak
    '''

    tpeak = np.apply_along_axis(np.argmax, 0, output_array)
    tpeak = tpeak * dt 

    return(tpeak)

#Value of hydrograph peak
def hydgrph_3D_peakval(output_array, dt=1):
    '''
        Calculate the time to peak for hydrographs 

        Assumes that data are generated in 3D arrays where the first dimension is time such that
        sumulated_output[:,j,i] would be the timeseries hydrograph at a single point

        Returns a spatial map of values with the max value for each grid cell. 
        
        Parameters:
            output_array (nparray) : (nt, n_dim2, n_dim2) array of values
            dt (nparray) : the timestep for the simulation, default = 1
        Returns:
            max_val(nparray): n_dim2 by n_dim1 array of the hydrograph maximum
    '''

    max_val = np.apply_along_axis(np.max, 0, output_array)

    return(max_val)

def hydgrph_3D_total(output_array, dt=1):
    '''
        Calculate the sum of a hydrograph

        Assumes that data are generated in 3D arrays where the first dimension is time such that
        sumulated_output[:,j,i] would be the timeseries hydrograph at a single point

        Returns a spatial map of values with the total flow
        
        Parameters:
            output_array (nparray) : (nt, n_dim2, n_dim2) array of values
            dt (nparray) : the timestep for the simulation, default = 1
        Returns:
            tprsk (nparray): n_dim2 by n_dim1 array of the time to peak
    '''

    htotal = np.apply_along_axis(np.sum, 0, output_array)
    htotal = htotal * dt

    return(htotal)

# cumulative dif
def hydgrph_3D_cumdiff(simulated_output, target_output):
    '''
        Calculate the total difference in between hydrograph outflow 

        Assumes that data are generated in 3D arrays where the first dimension is time such that
        sumulated_output[:,j,i] would be the timeseries hydrograph at a single point
        
        Returns a 2D map of cumulative differences for every gridcell

        Parameters:
            simulated_output (nparray) : (nt, n_dim2, n_dim2) array of outputs from the ML simulation
            target_output (nparray) : (nt, n_dim2, n_dim2) array of target outputs
        Returns:
            dif_map (nparray): n_dim2 by n_dim1 array of the total hydrograph differences
    '''

    n_dim2 = simulated_output.shape[1]
    n_dim1 = target_output.shape[2]

    dif_map = np.zeros((n_dim2, n_dim1))
    for jj in range(n_dim2):
        for ii in range(n_dim1):
            dif = simulated_output[:, jj, ii] - target_output[:, jj, ii]
            dif_map[jj, ii] = np.sum(dif)

    return(dif_map)
