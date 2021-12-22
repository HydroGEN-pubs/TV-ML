import json
import math
import numpy as np
import os
import sys

from parflow.tools import Run
from parflow.tools.fs import get_absolute_path
from parflowio.pyParflowio import PFData

# -----------------------------------------------------------------------------
# DataAccessor
# -----------------------------------------------------------------------------

class DataAccessor:

  def __init__(self, run, selector=None):
      self._run = run
      self._name = run.get_name()
      self._selector = selector
      self._t_padding = 5
      self._time = None
      self._ts = None
      # Initialize time
      self.time = 0

  # ---------------------------------------------------------------------------

  def _pfb_to_array(self, file_path):
    array = None
    if file_path:
      full_path = get_absolute_path(file_path)
      # FIXME do something with selector inside parflow-io
      pfb_data = PFData(full_path)
      pfb_data.loadHeader()
      pfb_data.loadData()
      array = pfb_data.moveDataArray()
      pfb_data.close()

    return array

  # ---------------------------------------------------------------------------
  # time
  # ---------------------------------------------------------------------------

  @property
  def time(self):
    return self._time

  @time.setter
  def time(self, t):
    self._time = int(t)
    self._ts = f'{self._time:0>{self._t_padding}}'

  @property
  def times(self):
    t0 = self._run.TimingInfo.StartCount
    t_start = self._run.TimingInfo.StartTime
    t_end = self._run.TimingInfo.StopTime
    t_step = self._run.TimeStep.Value
    t = t0 + t_start
    time_values = []
    keep_time_step = True
    while t <= t_end:
      time_values.append(int(t))
      t += t_step

    return time_values

  # ---------------------------------------------------------------------------
  # Region selector
  # ---------------------------------------------------------------------------

  @property
  def selector(self):
    return self._selector

  @selector.setter
  def set_selector(self, selector):
    self._selector = selector

  # ---------------------------------------------------------------------------
  # Grid information
  # ---------------------------------------------------------------------------

  @property
  def shape(self):
    # FIXME do something with selector
    return (
      self._run.ComputationalGrid.NZ,
      self._run.ComputationalGrid.NY,
      self._run.ComputationalGrid.NX
    )

  # ---------------------------------------------------------------------------
  # Mannings Roughness Coef  
  # ---------------------------------------------------------------------------

  @property
  def mannings(self):
    # @RMM added mannings roughness coeff
      return self._pfb_to_array(f'{self._run.get_name()}.out.mannings.pfb')
 

  # ---------------------------------------------------------------------------
  # Slopes X Y 
  # ---------------------------------------------------------------------------

  @property
  def slope_x(self):
    # @RMM check to see if slope X was input, otherwise fix for output filename
    if(self._run.TopoSlopesX.FileName== None):
      return self._pfb_to_array(f'{self._run.get_name()}.out.slope_x.pfb')
    else:
      return self._pfb_to_array(self._run.TopoSlopesX.FileName)

    

  @property
  def slope_y(self):
    # @RMM check to see if slope Y was input, otherwise fix for output filename
    if(self._run.TopoSlopesY.FileName== None):
      return self._pfb_to_array(f'{self._run.get_name()}.out.slope_y.pfb')
    else:
      return self._pfb_to_array(self._run.TopoSlopesY.FileName)


  @property
  def slope_z(self):  #@RMM no slope Z, we should remove
    return self._pfb_to_array(self._run.TopoSlopesZ.FileName)

  # ---------------------------------------------------------------------------
  # Computed Porosity
  # ---------------------------------------------------------------------------

  @property
  def computed_porosity(self):
    return self._pfb_to_array(f'{self._run.get_name()}.out.porosity.pfb')

  # ---------------------------------------------------------------------------
  # Computed Permeability
  # ---------------------------------------------------------------------------

  @property
  def computed_permeability_x(self):
    return self._pfb_to_array(f'{self._run.get_name()}.out.perm_x.pfb')

  @property
  def computed_permeability_y(self):
    return self._pfb_to_array(f'{self._run.get_name()}.out.perm_y.pfb')

  @property
  def computed_permeability_z(self):
    return self._pfb_to_array(f'{self._run.get_name()}.out.perm_z.pfb')

  # ---------------------------------------------------------------------------
  # Pressures
  # ---------------------------------------------------------------------------

  @property
  def pressure_initial_condition(self):
    press_type = self._run.ICPressure.Type
    if press_type == 'PFBFile':
      geom_name = self._run.ICPressure.GeomNames
      if len(geom_name) > 1:
        raise Exception(f'ICPressure.GeomNames are set to {geom_name}')
      return self._pfb_to_array(self._run.Geom[geom_name[0]].ICPressure.FileName)
    else:
      # HydroStaticPatch, ... ?
      raise Exception(f'Initial pressure of type {press_type} is not supported')

    return None

  # ---------------------------------------------------------------------------

  @property
  def pressure_boundary_conditions(self):
    # Extract all BC names (bc[{patch_name}__{cycle_name}] = value)
    bc = {}
    patch_names = []

    # Handle patch names
    if patch_name is not None:
      patch_names.append(patch_name)
    else:
      main_name = self._run.Domain.GeomName
      all_names = self._run.Geom[main_name].Patches
      patch_names.extend(all_names)

    # Extract cycle names for each patch
    for p_name in patch_names:
      cycle_name = self._run.Patch[p_name].BCPressure.Cycle
      cycle_names = self._run.Cycle[cycle_name].Names
      for c_name in cycle_names:
        key = f'{p_name}__{c_name}'
        bc[key] = self._run.Patch[p_name].BCPressure[c_name].Value

    return bc

  # ---------------------------------------------------------------------------

  @property
  def pressure(self):
    file_name = get_absolute_path(f'{self._name}.out.press.{self._ts}.pfb')
    return self._pfb_to_array(file_name)

  # ---------------------------------------------------------------------------
  # Saturations
  # ---------------------------------------------------------------------------

  @property
  def saturation(self):
    file_name = get_absolute_path(f'{self._name}.out.satur.{self._ts}.pfb')
    return self._pfb_to_array(file_name)

# -----------------------------------------------------------------------------
# ML Batch Generator
# -----------------------------------------------------------------------------

class Channel():
  def __init__(self, array_builder):
    self._array_builder = array_builder

  def __enter__(self):
    self._array_builder._channel_count += 1
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    pass

  def set(self, array):
    self._array_builder.add_input(array)

# -----------------------------------------------------------------------------

class Label():
  def __init__(self, array_builder):
    self._array_builder = array_builder

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    pass

  def set(self, array):
    self._array_builder.add_label(array)

# -----------------------------------------------------------------------------

class Batch():
  def __init__(self, array_builder):
    self._array_builder = array_builder

  def __enter__(self):
    self._array_builder._batch_size += 1
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    # Make sure the channels are valid
    if self._array_builder._channel_size:
      if self._array_builder._channel_size != self._array_builder._channel_count:
        raise Exception(f'Closing batch with {self._array_builder._channel_count} channels while expecting {self._array_builder._channel_size}')
    else:
      self._array_builder._channel_size = self._array_builder._channel_count

    self._array_builder._channel_count = 0

# -----------------------------------------------------------------------------

def add_array(array, name, t_shape, t_size, t_dim, t_fill):
  if t_size:
    # Validate size to be sure
    if array.size != t_size:
      raise Exception(f'Provided {name} does not have the proper size {array.shape}/{array.size} while expecting {t_shape}/{t_size}')
  else:
    # We need to capture the expected size
    t_size = array.size
    full_shape = array.shape
    compact_shape = []
    if len(full_shape) == t_dim:
      t_shape = full_shape
    elif t_dim == 1:
      t_shape = [t_size]
    else:
      # One or more dimension must be 1
      for dim in full_shape:
        if dim > 1:
          compact_shape.append(dim)

      # Ensure target dimension is matching target by appending dim(1) on the front
      while len(compact_shape) < t_dim:
        compact_shape.insert(0, 1)

      # Ensure target dimension match our array
      if len(compact_shape) != t_dim:
        raise Exception(f'Provided channel does not have the expected dimension of {t_dim} with shape {array.shape} => {compact_shape}')

      t_shape = compact_shape

  t_fill.append(array)
  return t_shape, t_size

# -----------------------------------------------------------------------------

class MLArrayBuilder:
  def __init__(self, input_dimension=1, output_dimension=1):
    self._input_dimension = input_dimension
    self._label_dimension = output_dimension
    self._input_shape = None
    self._input_size = 0
    self._label_shape = None
    self._label_size = 0
    self._inputs = []
    self._labels = []
    self._batch_size = 0
    self._channel_size = 0
    self._channel_count = 0

  def batch(self):
    return Batch(self)

  def channel(self):
    return Channel(self)

  def label(self):
    return Label(self)

  def clear(self):
    self._input_size = 0
    self._label_size = 0
    self._inputs = []
    self._labels = []
    self._batch_size = 0
    self._channel_size = 0
    self._channel_count = 0

  def add_input(self, array):
    self._input_shape, self._input_size = add_array(
      array,
      'channel',
      self._input_shape, self._input_size, self._input_dimension,
      self._inputs
    )

  def add_label(self, array):
    self._label_shape, self._label_size = add_array(
      array,
      'label',
      self._label_shape, self._label_size, self._label_dimension,
      self._labels
    )

  @property
  def inputs(self):
    # [batch, channel, ...(channel_shape)]
    end_shape = (self._batch_size, self._channel_size, *self._input_shape)
    return np.concatenate(self._inputs, axis=None).reshape(end_shape)

  @property
  def labels(self):
    # [batch, ...(label_shape)]
    end_shape = (self._batch_size, *self._label_shape)
    return np.concatenate(self._labels, axis=None).reshape(end_shape)
