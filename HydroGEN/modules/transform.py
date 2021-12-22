import numpy as np

def float32_clamp_scaling(src_range=[0, 50], dst_range=[-1, 1], print_range=False):
  in_delta = src_range[1] - src_range[0]
  in_min = src_range[0]
  in_max = src_range[1]
  out_min = dst_range[0]
  out_max = dst_range[1]
  out_delta = dst_range[1] - dst_range[0]
  out_scale = out_delta / in_delta

  def convert(array):
      min_value = np.amin(array)
      max_value = np.amax(array)
      
      out = np.empty(array.shape, dtype=np.float32)
      out= (array - in_min) * out_scale + out_min
      out[np.where(array<in_min)]=out_min
      out[np.where(array>in_max)]=out_max
      
      if print_range:
        print(f'Array range[{min_value}, {max_value}]')
      return out

  return convert

# -----------------------------------------------------------------------------
# Transform with boundary add-on
# -----------------------------------------------------------------------------

def float32_clamp_scaling_x_bc(src_range=[0, 50], dst_range=[-1, 1], height=50):
  in_delta = src_range[1] - src_range[0]
  in_min = src_range[0]
  in_max = src_range[1]
  out_min = dst_range[0]
  out_max = dst_range[1]
  out_delta = dst_range[1] - dst_range[0]
  out_scale = out_delta / in_delta
  bc_left = np.empty(height, dtype=np.float32)
  bc_right = np.empty(height, dtype=np.float32)

  def set_left(value):
    for i in range(height):
      if i < value:
        bc_left[i] = dst_range[1]
      else:
        bc_left[i] = dst_range[0]

  def set_right(value):
    for i in range(height):
      if i < value:
        bc_right[i] = dst_range[1]
      else:
        bc_right[i] = dst_range[0]

  def convert(array):
    in_shape = array.shape

    # convert 3D => 2D
    if len(in_shape) == 3:
      in_shape = (in_shape[0], in_shape[2])
      array = array.reshape(in_shape)

    out_shape = (in_shape[0], in_shape[1] + 2)
    out = np.empty(out_shape, dtype=np.float32)
    # print(f'in shape {in_shape} => out shape {out_shape} => real {out.shape}')
    for z in range(in_shape[0]):
      for x in range(in_shape[1]):
        value = array[z,x]
        if value < in_min:
          out[z, x + 1] = out_min
        elif value > in_max:
          out[z, x + 1] = out_max
        else:
          out[z, x + 1] = (value - in_min) * out_scale + out_min

    # Add BCs
    for i in range(height):
      out[i, 0] = bc_left[i]
      out[i, out_shape[1] - 1] = bc_right[i]

    return out

  return convert, set_left, set_right
