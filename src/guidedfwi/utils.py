import torch
import numpy as np
import random
import dask.array as da

from skimage.transform import resize
from os import listdir, path
from shutil import rmtree
from tempfile import gettempdir
from scipy.fft import fft, ifft, fftfreq
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

def extract_cubes_fast(cube, cube_size, stride):
    """
    Extracting cubes from an array of size DxWxH
    """
    
    # Get the shape of the input cube
    d, h, w = cube.shape

    # Calculate the number of smaller cubes in each dimension
    num_cubes_d = (d - cube_size[0]) // stride[0] + 1
    num_cubes_h = (h - cube_size[1]) // stride[1] + 1
    num_cubes_w = (w - cube_size[2]) // stride[2] + 1

    # Initialize the output array for smaller cubes
    output = np.empty((num_cubes_d * num_cubes_h * num_cubes_w, cube_size[0], cube_size[1], cube_size[2]), dtype=cube.dtype)

    # Define a helper function for cube extraction
    def extract_cube(i, j, k):
        return cube[i:i+cube_size[0], j:j+cube_size[1], k:k+cube_size[2]]

    # Extract smaller cubes in parallel
    index = 0
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, d - cube_size[0] + 1, stride[0]):
            for j in range(0, h - cube_size[1] + 1, stride[1]):
                for k in range(0, w - cube_size[2] + 1, stride[2]):
                    futures.append(executor.submit(extract_cube, i, j, k))

        # Retrieve the extracted cubes from the futures
        for future in futures:
            output[index] = future.result()
            index += 1

    return output

def normalize_to_zero_and_one(v, vmin=1500.0, vmax=4500.0):
    """
    Normalize from physical velocity range [vmin, vmax] to [-1, 1]
    """
    return (v - vmin) / (vmax - vmin)

def denormalize_from_zero_and_one(x, vmin=1500.0, vmax=4500.0):
    """
    Denormalize from [-1, 1] back to physical velocity range [vmin, vmax]
    """
    return x  * (vmax - vmin) + vmin

def normalize_to_minusone_and_one(v, vmin=1500.0, vmax=4500.0):
    """
    Normalize from physical velocity range [vmin, vmax] to [-1, 1]
    """
    return 2 * (v - vmin) / (vmax - vmin) - 1

def denormalize_from_minusone_and_one(x, vmin=1500.0, vmax=4500.0):
    """
    Denormalize from [-1, 1] back to physical velocity range [vmin, vmax]
    """
    return ((x + 1) / 2) * (vmax - vmin) + vmin

def extract_windows(dask_array, window_size=(256, 256), stride=(128, 128)):
    """
    Extract 2D windows from a 3D Dask array.

    Parameters
    ----------
    - dask_array: A Dask array of shape (I, X, D).
    - window_size: Size of the window (height, width).
    - stride: Stride for moving the window (vertical, horizontal).

    Returns
    -------
    - A Dask array of extracted windows.
    """
    I, X, D = dask_array.shape
    window_height, window_width = window_size
    stride_h, stride_w = stride

    # Calculate the number of windows to extract
    if stride_h==0:
        n_windows_h=1
    else:
        n_windows_h = (X - window_height) // stride_h + 1
    
    if stride_w==0:
        n_windows_w=1
    else:
        n_windows_w = (D - window_width) // stride_w + 1

    # Extract windows using Dask
    windows = []

    for i in range(I):
        for h in range(n_windows_h):
            for w in range(n_windows_w):
                # Calculate the starting indices for the window
                start_h = h * stride_h
                start_w = w * stride_w
                # Extract the window and append to the list
                window = dask_array[i, start_h:start_h + window_height, start_w:start_w + window_width]
                windows.append(window)
    
    for x in range(X):
        for h in range(n_windows_h):
            for w in range(n_windows_w):
                # Calculate the starting indices for the window
                start_h = h * stride_h
                start_w = w * stride_w
                # Extract the window and append to the list
                window = dask_array[start_h:start_h + window_height, x, start_w:start_w + window_width]
                windows.append(window)

    # Stack the windows into a Dask array
    result = da.stack(windows)
    
    return result

def pad_dask_array_to_target_size(dask_array, target_shape):
    """
    Pad a 3D Dask array to the specified target size for each dimension.

    Parameters
    ----------
    - dask_array: A 3D Dask array.
    - target_shape: A tuple representing the target shape (new_depth, new_height, new_width).

    Returns
    -------
    - A new Dask array padded to the specified target size.
    """
    original_shape = dask_array.shape
    print(f"Original shape: {original_shape}, Target shape: {target_shape}")

    # Validate target shape
    if len(target_shape) != 3:
        raise ValueError("Target shape must be a tuple of three dimensions.")

    # Calculate padding widths for each dimension
    pad_widths = []
    for orig_size, target_size in zip(original_shape, target_shape):
        if target_size < orig_size:
            raise ValueError("Target size must be greater than or equal to the original size.")
        
        total_pad = target_size - orig_size
        # Distribute padding evenly (or overflow to the end if odd)
        pad_widths.append((total_pad // 2, total_pad - (total_pad // 2)))

    # Use da.pad to add padding, filling with the edge values of the original array
    padded_array = da.pad(dask_array, pad_widths, mode='edge')

    return padded_array

def set_seed(seed):
    """
    An integer of random number.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def highpass_filter(signal, dt, cutoff_freq):
    """
    Perform high-pass filter.
    """

    # Perform Fourier transform
    fft_signal = fft(signal)
    freqs = fftfreq(len(signal), d=dt)
    
    # Create a high-pass filter
    filter_mask = np.abs(freqs) >= cutoff_freq
    
    # Apply the filter
    fft_signal_filtered = fft_signal * filter_mask
    
    # Inverse Fourier transform to obtain the filtered signal
    filtered_signal = ifft(fft_signal_filtered)
    return filtered_signal.real

def save_dict_as_compressed_npz(file_name, data_dict):
    """
    Save a dictionary with array values to a compressed .npz file, ensuring all data is in float32 format.
    Converts PyTorch tensors to NumPy arrays and casts to float32 automatically.

    Parameters
    ----------
     file_name: str, the name of the file to save the data to.
     data_dict: dict, a dictionary with values as arrays (NumPy or PyTorch tensors).
    """
    # Prepare data for saving: convert PyTorch tensors to NumPy arrays and cast to float32 if necessary
    save_dict = {}
    for key, value in data_dict.items():
        # Handle PyTorch tensors
        if isinstance(value, torch.Tensor):
            converted_value = value.cpu().numpy().astype(np.float32)
        # Handle NumPy arrays and cast to float32 if not already
        elif isinstance(value, np.ndarray) and value.dtype != np.float32:
            converted_value = value.astype(np.float32)
        else:
            converted_value = value  # For non-numeric types, no conversion applied

        save_dict[key] = converted_value

    # Save as a compressed .npz file, using the dictionary keys as variable names
    np.savez_compressed(file_name, **save_dict)

def load_bin(p, dims):
    """
    Load binary files
    """
    f = open(p)
    vp = np.fromfile (f, dtype=np.dtype('float32').newbyteorder ('<'))
    f.close()
    vp = vp.reshape(*dims)
    vp = np.transpose(vp)
    vp = np.flipud(vp)
    return vp

def clear_devito_cache():
    """
    Clear devito cache
    """
    tempdir = gettempdir()
    for i in listdir(tempdir):
        if i.startswith('devito-'):
            try:
                target = path.join(tempdir, i)
                rmtree(target)
            except:
                pass

def numpy_to_cuda(x):
    return torch.from_numpy(x).cuda()

def cuda_to_numpy(x):
    return x.detach().cpu().numpy()

def extract_squares(array, square_size, stride):
    """
    Extract smaller patches from 2D array.
    
    Parameters
        array (array): _description_
        square_size (floats): _description_
        stride (floats): _description_

    Returns
        list: Contains smaller image patches
    """
    dx, dy = square_size
    sx, sy = stride
    squares = []

    # Pad the array using edge values
    pad_x = (0, (dx - array.shape[0] % dx) % dx)
    pad_y = (0, (dy - array.shape[1] % dy) % dy)
    padded_array = np.pad(array, (pad_x, pad_y), mode='edge')

    for x in range(0, padded_array.shape[0] - dx + 1, sx):
        for y in range(0, padded_array.shape[1] - dy + 1, sy):
            square = padded_array[x:x + dx, y:y + dy]
            squares.append(square)

    return squares

def combine_squares(squares, original_shape, square_size, stride):
    """
    Combine smaller patches into their original shape.

    Parameters
        squares (_type_): _description_
        original_shape (_type_): _description_
        square_size (_type_): _description_
        stride (_type_): _description_

    Returns
        array: Combined patches.
    """
    dx, dy = square_size
    sx, sy = stride

    combined = np.zeros((original_shape[0] + (dx - original_shape[0] % dx) % dx, 
                         original_shape[1] + (dy - original_shape[1] % dy) % dy))
    weight = np.zeros_like(combined)

    index = 0
    for x in range(0, combined.shape[0] - dx + 1, sx):
        for y in range(0, combined.shape[1] - dy + 1, sy):
            combined[x:x + dx, y:y + dy] += squares[index]
            weight[x:x + dx, y:y + dy] += 1
            index += 1

    # Avoid division by zero
    weight[weight == 0] = 1

    # Trim the padding to return the original shape
    combined = combined[:original_shape[0], :original_shape[1]]
    weight = weight[:original_shape[0], :original_shape[1]]

    return combined / weight

class NumpyDataset(Dataset):
    def __init__(self, file_path=None, folder_path=None, cube_size=None, channels=1, normalize=False):
        """
        Initializes the dataset by loading numpy files from the specified folder.

        :param folder_path: Path to the folder containing .npy files
        """
        self.folder_path = folder_path
        self.data = []
        self.idxs = []
        self.cube_size = cube_size

        if folder_path is not None:
            self.files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            for file in self.files:

                file_path = os.path.join(folder_path, file)
                cubes = np.load(file_path).astype(np.float16)

                # Reshape cubes to (B, 64, 64, 64)
                reshaped_cubes = self._reshape_cubes(cubes)
                self.data.append(reshaped_cubes)
                self.idxs.append(torch.ones(reshaped_cubes.shape[0]) * int(file_path.split('/')[-1].split('_')[0]))
                
                if normalize:
                    cubes = self._normalize(reshaped_cubes)

            # Concatenate all data tensors into a single tensor
            self.data = torch.tensor(np.concatenate(self.data, axis=0), dtype=torch.float32)
            self.idxs = torch.tensor(np.concatenate(self.idxs, axis=0), dtype=torch.float32)
            
        else:
            if self.cube_size is not None:
                self.data = torch.tensor(np.load(file_path).astype(np.float32), dtype=torch.float32).reshape(-1,channels,self.cube_size,self.cube_size,self.cube_size)
            else:
                self.data = torch.tensor(np.load(file_path).astype(np.float32), dtype=torch.float32).unsqueeze(1)

    def _reshape_cubes(self, cubes):
        """
        Reshape the numpy array of cubes to (B, 64, 64, 64).

        :param cubes: Numpy array of shape (B, D1, D2, D3)
        :return: Reshaped numpy array
        """
        _, D, _, _ = cubes.shape
        
        # Assuming D1, D2, D3 are multiples of 64
        multiplier = D // self.cube_size

        return cubes[:, ::multiplier, ::multiplier, ::multiplier]

    def __len__(self):
        """
        Returns the total number of cubes in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns the cube at the specified index.

        :param idx: Index of the cube to retrieve
        :return: Tensor cube of shape (64, 64, 64)
        """
        return self.data[idx]

    def _normalize(self, x):
        return 2*((x-x.min())/(x.max()-x.min())) - 1

def extract_cubes(array, cube_size, stride):
    """
    Extract overlapping small cubes from a 3D array.

    Parameters
    ----------
    array (np.ndarray): The input 3D array.
    cube_size (tuple): The size of each cube (dx, dy, dz).
    stride (tuple): The stride between cubes (sx, sy, sz).

    Returns
    -------
    list: A list of extracted cubes.
    """
    dx, dy, dz = cube_size
    sx, sy, sz = stride
    cubes = []

    for x in range(0, array.shape[0] - dx + 1, sx):
        for y in range(0, array.shape[1] - dy + 1, sy):
            for z in range(0, array.shape[2] - dz + 1, sz):
                cube = array[x:x + dx, y:y + dy, z:z + dz]
                cubes.append(cube)

    return cubes

def combine_cubes(cubes, original_shape, cube_size, stride):
    """
    Combine small cubes back into the original 3D array.

    Parameters
    ----------
    cubes (list): A list of extracted cubes.
    original_shape (tuple): The shape of the original 3D array.
    cube_size (tuple): The size of each cube (dx, dy, dz).
    stride (tuple): The stride between cubes (sx, sy, sz).

    Returns
    -------
    np.ndarray: The reconstructed 3D array.
    """
    dx, dy, dz = cube_size
    sx, sy, sz = stride

    combined = np.zeros(original_shape)
    weight = np.zeros(original_shape)

    index = 0
    for x in range(0, original_shape[0] - dx + 1, sx):
        for y in range(0, original_shape[1] - dy + 1, sy):
            for z in range(0, original_shape[2] - dz + 1, sz):
                combined[x:x + dx, y:y + dy, z:z + dz] += cubes[index]
                weight[x:x + dx, y:y + dy, z:z + dz] += 1
                index += 1

    # Avoid division by zero
    weight[weight == 0] = 1

    return combined / weight

class NumpyCubeDataset(Dataset):
    def __init__(self, file_path=None, folder_path=None, cube_size=64, channels=1, normalize=False):
        """
        Initializes the dataset by loading numpy files from the specified folder.

        :param folder_path: Path to the folder containing .npy files
        """
        self.folder_path = folder_path
        self.data = []
        self.idxs = []
        self.cube_size = cube_size

        if folder_path is not None:
            self.files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            for file in self.files:

                file_path = os.path.join(folder_path, file)
                cubes = np.load(file_path).astype(np.float16)

                # Reshape cubes to (B, 64, 64, 64)
                reshaped_cubes = self._reshape_cubes(cubes)
                self.data.append(reshaped_cubes)
                self.idxs.append(torch.ones(reshaped_cubes.shape[0]) * int(file_path.split('/')[-1].split('_')[0]))
                
                if normalize:
                    cubes = self._normalize(reshaped_cubes)

            # Concatenate all data tensors into a single tensor
            self.data = torch.tensor(np.concatenate(self.data, axis=0), dtype=torch.float32)
            self.idxs = torch.tensor(np.concatenate(self.idxs, axis=0), dtype=torch.float32)
            
        else:
            self.data = torch.tensor(np.load(file_path).astype(np.float32), dtype=torch.float32).reshape(-1,channels,self.cube_size,self.cube_size,self.cube_size)

    def _reshape_cubes(self, cubes):
        """
        Reshape the numpy array of cubes to (B, 64, 64, 64).

        :param cubes: Numpy array of shape (B, D1, D2, D3)
        :return: Reshaped numpy array
        """
        _, D, _, _ = cubes.shape
        
        # Assuming D1, D2, D3 are multiples of 64
        multiplier = D // self.cube_size

        return cubes[:, ::multiplier, ::multiplier, ::multiplier]

    def __len__(self):
        """
        Returns the total number of cubes in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns the cube at the specified index.

        :param idx: Index of the cube to retrieve
        :return: Tensor cube of shape (64, 64, 64)
        """
        return self.data[idx]

    def _normalize(self, x):
        return 2*((x-x.min())/(x.max()-x.min())) - 1
    
import os
import time
import yaml
import sys
import platform
import psutil
import inspect
    
def log_experiment(results_dir, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), config_file=None, run_id=time.strftime("%Y%m%d-%H%M%S")):
    """
    Logs experiment details to a text file.
    
    Parameters
    ----------
    config_file (str): A yaml configufation file directory.
    results_dir (str): A string of results directory.

    Return:
    ----------
    log (file): Experiment summary.

    """
    
    log_file = os.path.join(results_dir, f"{run_id}.log")

    command = " ".join(sys.argv)
    environment = dict(os.environ)

    # Get system information
    system_info = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "ram_total": psutil.virtual_memory().total,
    }

    # Get GPU information (if available)
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            }
    except ImportError:
        pass

    # Get the current script's source code
    with open(script_path, 'r') as script_file:
        script_code = script_file.read()

    with open(log_file, 'w') as lf:
        lf.write("#########################################################\n")
        lf.write("### Experiment Summary ###\n")
        lf.write("#########################################################\n\n")

        lf.write(f"Run ID (YearMonthDate-HourMinuteSecond): {run_id}\n\n")
        lf.write(f"Command Line: {command}\n\n")
        
        lf.write("Environment Variables: \n")
        for key, value in environment.items():
            lf.write(f"  {key}: {value}\n")
        lf.write("\n")

        lf.write("System Information:\n")
        for key, value in system_info.items():
            lf.write(f"  {key}: {value}\n")
        lf.write("\n")

        if gpu_info:
            lf.write("GPU Information:\n")
            for key, value in gpu_info.items():
                lf.write(f"  {key}: {value}\n")
            lf.write("\n")

        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            lf.write("#########################################################\n")
            lf.write("### YAML Configuration ###\n")
            lf.write("#########################################################\n\n")
            lf.write(yaml.dump(config, indent=2))
            lf.write("\n")

        lf.write("#########################################################\n")
        lf.write("### Script ###\n")
        lf.write("#########################################################\n\n")

        lf.write(script_code)
        lf.write("\n")

        lf.write("#########################################################\n")