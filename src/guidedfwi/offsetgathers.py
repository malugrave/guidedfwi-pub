from devito import Eq, Operator, Function, TimeFunction, solve
from examples.seismic.utils import sources, PointSource

def generate_index_pairs(array_length, window_length, num_windows):
    """
    Generate pairs of indices for windows of a specified length over an array of a given length.

    Parameters
    ----------
    - array_length: Total length of the array.
    - window_length: Length of each window.
    - num_windows: Number of windows to generate.

    Returns
    -------
    - A list of tuples representing the start and end indices of each window.
    """
    if window_length * num_windows > array_length:
        raise ValueError("Total length of windows exceeds array length.")
    
    step = (array_length - window_length) // (num_windows - 1) if num_windows > 1 else 0
    index_pairs = []

    for n in range(num_windows):
        start = n * step
        end = start + window_length
        index_pairs.append((start, end - 1))
    
    return index_pairs

def zero_outside_range(arr, index_range, axis=0):
    """
    Zero out elements outside the given index range along the specified axis in a 2D NumPy array.

    Parameters
    ----------
    - arr: 2D numpy array to be modified.
    - index_range: A tuple or list specifying the inclusive range (start, end).
    - axis: Axis along which to apply the range (0 for rows, 1 for columns).

    Returns
    -------
    - A new 2D numpy array with elements zeroed out outside the index range.
    """
    start, end = index_range
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    result = arr.copy()  # Make a copy of the array to avoid modifying the original

    if axis == 0:
        # Zero out rows outside the range
        result[:start, :] = 0
        result[end+1:, :] = 0
    elif axis == 1:
        # Zero out columns outside the range
        result[:, :start] = 0
        result[:, end+1:] = 0

    return result

def ImagingOperator(model, image, space_order, geometry):
    # Define the wavefield with the size of the model and the time dimension
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order,
                     save=geometry.nt)
    
    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward))
    
    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)    
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)
    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image + u * v)


    return Operator([stencil] + res_term + [image_update],
                    subs=model.spacing_map)