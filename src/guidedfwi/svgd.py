import numpy as np
import torch
import gc
import deepwave 
import math
from scipy.ndimage import gaussian_filter

from .plots import plot_flipped_data, plot_modulus

def compute_gradient(
    model,
    x_s,
    x_r,
    nz,
    nx,
    dx,
    dt,
    freq,
    data_true,
    source_wavelet,
    log_likelihood,
    batch_size,
    data_normalization=None,
    device="cpu",
    weights=None,
    grad_max_ref=1,
    vp_mask=None,
    smooth_sigma=0,
    vmin=1500,
    vmax=4500,
    receiver_mask=None
):
    """
    Compute the FWI gradient of the loss function in mini-batches using Deepwave,
    with optional normalization and masking.

    Parameters
    -----------
    model : Tensor
        Velocity model of shape [nz, nx] or flattened [nz * nx].
    x_s : Tensor
        Source coordinates [num_shots, 1, 2] (grid units).
    x_r : Tensor
        Receiver coordinates [num_shots, num_receivers, 2] (grid units).
    nz, nx : int
        Model dimensions.
    dx : list or tuple
        Grid spacing [dz, dx].
    dt : float
        Time step.
    freq : float
        PML frequency.
    data_true : Tensor
        Observed seismic data [num_shots, num_receivers, nt].
    source_wavelet : Tensor
        Source wavelet [num_shots, 1, nt].
    log_likelihood : Callable
        Loss function.
    batch_size : int
        Number of shots per batch.
    data_normalization : Callable or None
        Optional normalization for data.
    device : str
        Device to use.
    weights : list
        Weighting factors for prediction and truth in loss.
    grad_max_ref : float or None
        If provided, normalize gradients by this value (used in SVGD).
    vp_mask : Tensor or None
        Mask to apply to gradient (e.g., for water layer).
    smooth_sigma : float
        Gaussian smoothing standard deviation for gradient (0 disables).

    Returns
    --------
    running_loss : float
        Total accumulated loss.
    grad_loss : ndarray
        Processed gradient [nz, nx].
    grad_max_ref : float
        Maximum absolute gradient used for normalization.
    """
    model = model.reshape(nz, nx) if len(model.shape) == 1 else model
    model = model.to(device).requires_grad_(True)

    running_loss = 0.0
    grad_loss = torch.zeros_like(model)

    num_shots = x_s.shape[0]
    
            # for it in range(0, num_shots, num_batches):
            #     batch_src_amps = source_amplitudes[it:it+num_batches, :, :]
            #     batch_obs_data = obs_data[it:it+num_batches, :, :].to(device)[receiver_mask[it:it+num_batches, :]]
            #     batch_src_locs = source_locations[it:it+num_batches, :, :].to(device)
            #     batch_rcv_locs = receiver_locations[it:it+num_batches, :, :].to(device)
                
            #     batch_syn_data = deepwave.scalar(
            #         vp_smooth,
            #         grid_spacing=[dz, dx],
            #         dt=dt,
            #         source_amplitudes=batch_src_amps,
            #         source_locations=batch_src_locs,
            #         receiver_locations=batch_rcv_locs,
            #         accuracy=8,
            #         pml_freq=freq,
            #         pml_width=[0, 10, 10, 10],
            #         max_vel=4500
            #     )[-1]
                
            #     batch_syn_data /= scaler
            #     batch_syn_data = batch_syn_data[receiver_mask[it:it+num_batches, :].to(device)].to(device)

            #     mask = [torch.ones_like(batch_syn_data), torch.ones_like(batch_syn_data)]

            #     loss = loss_function(
            #         obs_weight * (batch_syn_data) * mask[0],
            #         syn_weight * (batch_obs_data) * mask[1],
            #     )
            #     loss.backward() #retain_graph=True)
            #     running_loss += loss.item()

    for it in range(0, num_shots, batch_size):
        batch_src_wvl = source_wavelet[it:it+batch_size].to(device)
        batch_data_true = data_true[it:it+batch_size].to(device)
        batch_x_s = x_s[it:it+batch_size].to(device)
        batch_x_r = x_r[it:it+batch_size].to(device)

        data_pred = deepwave.scalar(
            model, dx, dt,
            source_amplitudes=batch_src_wvl,
            source_locations=batch_x_s,
            receiver_locations=batch_x_r,
            accuracy=8,
            pml_freq=freq,
            pml_width=[0, 10, 10, 10],
            max_vel=vmax
        )[-1]

        if data_normalization is not None:
            batch_data_true = data_normalization(batch_data_true)
            data_pred = data_normalization(data_pred)
            
        if receiver_mask is not None:
            batch_data_true = batch_data_true[receiver_mask[it:it+batch_size, :]]
            data_pred = data_pred[receiver_mask[it:it+batch_size, :]]
            
        if weights is not None:
            loss = log_likelihood(weights[0] * data_pred, weights[0] * batch_data_true)
        else:
            loss = log_likelihood(data_pred, batch_data_true)
        running_loss += loss.item()

        grad_loss += torch.autograd.grad(loss, model, retain_graph=True)[0]
        
    grad_loss /= grad_max_ref

    # Optional Gaussian smoothing
    if smooth_sigma > 0:
        grad_loss = torch.tensor(
            gaussian_filter(grad_loss.cpu().numpy(), smooth_sigma),
            device=device
        )

    # Apply water layer mask
    if vp_mask is not None:
        grad_loss *= vp_mask.to(device)
        assert vp_mask.shape == grad_loss.shape
        
    # Bounds projection and smoothing
    model.data[model.data < vmin] = vmin
    model.data[model.data > vmax] = vmax  

    gc.collect()
    torch.cuda.empty_cache()

    return running_loss, grad_loss.detach().cpu().numpy()

def compute_gradient_per_batch(model, gradient_function):
    """
    Compute FWI gradients for a batch of particles using SVGD.

    Parameters
    -----------
    model : Tensor
        Batch of velocity models of shape [B, nz, nx] or flattened [B, nz * nx].
    gradient_function : Callable
        Function that computes loss and gradient for a single model.

    Returns
    --------
    log_p : float
        Mean loss value across all models in the batch.
    fwi_grad : ndarray
        Gradient of the loss for each model, shape [B, nz * nx].
    """

    log_p = 0.0
    fwi_grad = np.zeros_like(model.detach().cpu().numpy())
    for i, m in enumerate(model):
        m.requires_grad_(True)
        loss, grad_m = gradient_function(m)
        fwi_grad[i] = grad_m.ravel()
        log_p += loss
    log_p /= len(model)
    return log_p, fwi_grad

def compute_max_gradient(grad):
    """
    Compute the maximum absolute gradient value.

    Parameters
    -----------
    grad : ndarray
        Gradient array of shape [nz, nx] or flattened.

    Returns
    --------
    max_val : float
        Maximum absolute value in the gradient.
    """

    return np.max(np.abs(grad))

def compute_max_gradient_per_batch(grad):
    """
    Compute the maximum gradient per model in a batch.

    Parameters
    -----------
    grad : ndarray
        Batch of gradients of shape [B, nz * nx].

    Returns
    --------
    max_vals : ndarray
        Maximum absolute gradient value per model, shape [B, 1].
    """

    assert len(grad.shape) == 2
    return np.max(np.abs(grad), axis=1, keepdims=True)

class SteinVariationalGradientDescent:
    def __init__(self, particles, K, alpha, optimizer, scheduler, device='cuda'):
        """
        Simulate SVGD for n number of particles

        gmax: max FWI gradient of the initial particles (numpy.ndarray or torch.Tensor)
        K: Kernel function (e.g., RBF or IMQ) taking two arguments and returning a matrix
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler (optional)
        compute_gradient_per_batch: Function to compute FWI gradient per batches
        filter_func: Function to filter the gradient
        nz: Model dimensions in z
        nx: Model dimensions in x
        device: The torch device on which computations will be performed
        """
        self.particles = particles
        self.K = K
        self.alpha = alpha
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    

        # Variables to retrieve later
        self.log_grad_p = None
        self.sigma = None
        self.K_XX = None
        self.gradients = None
        self.driving_force = None
        self.repulsive_force = None

    def phi(self, particles, log_grad_p, epoch, EMA):
        """
        Compute the Stein Gradient. The terms inside the square bracket above.

        X: n number of models (particles)
        """
        X = particles.detach().requires_grad_(True)
        X = particles.clone()
        
        self.log_grad_p = log_grad_p
        self.K_XX, der, self.sigma = self.K(X, EMA)
    
        self.driving_force = self.K_XX.mm(self.log_grad_p)
        self.repulsive_force = der
        
        alpha_iter = self.alpha[epoch]
        
        phi = alpha_iter*self.driving_force - self.repulsive_force
        
        return phi

    def step(self, X, log_grad_p, m_vmin, m_vmax, epoch, gmax=1, EMA=None):
        """
        Bound model to the limits
        m_vmin: minimum model value
        m_vmax: maximum model value
        """
        self.optimizer.zero_grad()
        X.grad = self.phi(X, log_grad_p, epoch, EMA)/gmax
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step(log_grad_p)

        with torch.no_grad():
            X.clamp_(m_vmin, m_vmax)
            # X.clamp_(m_vmin-0.75, m_vmax+0.75)

class InverseQuadratic(torch.nn.Module):
    """Inverse Quadratic Kernel (IQ) with beta = -1/2 and c =1 following
    measuring sample quality with kernels by Gorham & Mackey, 2020

    Parameters
    ----------
    sigma : :obj:`numpy.float`
        Bandwidth of the RBF kernel

    Returns
    -------
    grad : :obj:`numpy.ndarray`
        Gradient of size ``(nx, nz)``
        
    """
    def __init__(self, sigma=None):
        super(InverseQuadratic, self).__init__()
        self.sigma = sigma

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)
       
        # Apply the median
        if self.sigma is None:
            sigma = torch.median(dists)
            
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        else:
            sigma = self.sigma

        k = 1. / (1. + dists / sigma)
        dxkxy = .5 * k / (1. + dists / sigma)
        der = (d * dxkxy[:, :, None]).sum(axis=0) * 2. / sigma
        
        # Delete the intermediate variables to free memory
        del d, dists, dxkxy
        torch.cuda.empty_cache()  # Clear memory cache if necessary

        return k, der, torch.sqrt(sigma)

class InverseMultiQuadric(torch.nn.Module):
    """Inverse Multiquadric Kernel (IMQ) with beta = -1/2 and c =1 following
    measuring sample quality with kernels by Gorham & Mackey, 2020

    Parameters
    ----------
    sigma : :obj:`numpy.float`
        Bandwidth of the RBF kernel

    Returns
    -------
    grad : :obj:`numpy.ndarray`
        Gradient of size ``(nx, nz)``
        
    """
    def __init__(self, sigma=None):
        super(InverseMultiQuadric, self).__init__()
        self.sigma = sigma

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)
       
        # Apply the median
        if self.sigma is None:
            sigma = torch.median(dists)
            
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        else:
            sigma = self.sigma

        k = 1. / torch.sqrt(1. + dists / sigma)
        dxkxy = .5 * k / (1. + dists / sigma)
        der = (d * dxkxy[:, :, None]).sum(axis=0) * 2. / sigma
        
        # Delete the intermediate variables to free memory
        del d, dists, dxkxy
        torch.cuda.empty_cache()  # Clear memory cache if necessary

        return k, der, torch.sqrt(sigma)

class RadialBasisFunction(torch.nn.Module):
    """Initializes the RadialBasisFunction class

    Parameters
    ----------
    sigma : :obj:`numpy.float`
        Bandwidth of the RBF kernel
    power : :obj:`int`
        Controls the RBF kernel variance

    Returns
    -------
    grad : :obj:`numpy.ndarray`
        Gradient of size ``(nx, nz)``
        
    """
    def __init__(self, sigma=None, power=2):
        super(RadialBasisFunction, self).__init__()
        self.sigma = sigma
        self.power = power
        

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)

        if self.sigma is None:
            h = torch.median(dists) / (2 * math.log(X.size(0) + 1))
            sigma = math.sqrt(h)
        
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        
        else:
            sigma = self.sigma

        k = torch.exp(-dists / sigma**self.power / 2)
        der = (d * k[:, :, None]).sum(axis=0) / sigma**self.power
    
        return k, der, sigma