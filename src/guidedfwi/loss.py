__all__ = ["L2",
           "L2Torch", 
           "Empty", 
           "L2MultiSource"
           ]

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity, TorchOperator
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name
from .nonlinear import NonlinearOperator

class L2MultiSource(NonlinearOperator):
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm defined as: :math:`\ell_2(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    size : :obj:`int`, optional
        Size of the input vector (needed only when both ``Op`` and
        ``b`` are ``None``)
    
    """
    def __init__(self, b, encoder=None, sigma=1., size=None, image=None, dtype="float32"):
        self.original_b = b.copy()  # <-- store unencoded b
        self.b = b.copy() # this will be overwritten on encode
        self.encoder = encoder
        self.sigma = sigma
        if size is None:
            size = b.size
        self.image = image
        super().__init__(size, dtype)
    def set_encoder(self, encoder):
        """Update the encoder without reinitializing the object."""
        self.encoder = encoder
    
    def encode_data(self):
        """Apply encoding to the original (unencoded) b, not the possibly re-encoded one."""
        if self.encoder is not None:
            self.b = self.encoder.apply(self.original_b, composite=False)


    def loss(self, x, i):
        self.bs = self.b[i].sum(0).ravel()
        f = (self.sigma / 2.) * (np.linalg.norm(x - self.bs) ** 2)


        return f
    
    def grad(self, x, i):
        g = self.sigma * (x - self.bs)
        return g

class Empty():
    r"""Empty Norm.
    
    This class implements an empty norm, where the gradient 
    is simply represented by the data vector (and the norm
    itself simply returns 0). To be used only to perform RTM
    (i.e., to have an adjoint source equal to the data)

    Parameters
    ----------
    b : :obj:`numpy.ndarray`
        Data vector
    
    """
    def __init__(self, b):
        self.b = b
        
    def __call__(self, x, i):
        return 0.
    
    def grad(self, x, i):
        return self.b[i]


class L2():
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm defined as: :math:`\ell_2(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    
    """
    def __init__(self, Op=None, b=None):
        self.Op = Op
        self.b = b
        
    def __call__(self, x, i):
        if self.Op is not None and self.b is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            f = (1. / 2.) * (np.linalg.norm(Op @ x - self.b[i]) ** 2)
        elif self.b is not None:
            f = (1. / 2.) * (np.linalg.norm(x - self.b[i]) ** 2)
        else:
            f = (1. / 2.) * (np.linalg.norm(x) ** 2)
        return f
    
    def grad(self, x, i):
        if self.Op is not None and self.b is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            g = Op.H @ (Op @ x - self.b[i])
        elif self.b is not None:
            g = (x - self.b[i])
        else:
            g = x
        return g

class L2Multi():
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm for multiple recordings defined as: :math:`\ell_2(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    c : :obj:`numpy.ndarray`, optional
        Data vector
    idx : :obj:`numpy.ndarray`, optional
        Source index
    
    """
    def __init__(self, Op=None, b=None, c=None, d=None):
        self.Op = Op
        self.b = b
        self.c = c
        self.d = d
        
    def __call__(self, x, y, idx):
        if self.Op is not None and self.b is not None:
            Op = self.Op[idx] if isinstance(self.Op, list) else self.Op
            f = (1. / 4.) * (np.linalg.norm(Op @ x - self.b[idx]) ** 2) + (3. / 4.) * (np.linalg.norm(Op @ y - self.c[idx]) ** 2)
        elif self.b is not None:
            f = (1. / 4.) * (np.linalg.norm(x - self.b[idx]) ** 2) + (3. / 4.) * (np.linalg.norm(y - self.c[idx]) ** 2)
        else:
            f = (1. / 4.) * (np.linalg.norm(x) ** 2) + (3. / 4.) * (np.linalg.norm(y) ** 2)
        return f
    
    def grad(self, x, y, idx):
        if self.Op is not None and self.b is not None:
            Op = self.Op[idx] if isinstance(self.Op, list) else self.Op
            g = Op.H @ (Op @ x - self.b[idx])
            h = Op.H @ (Op @ y - self.c[idx])
        elif self.b is not None:
            g = (x - self.b[idx])
            h = (y - self.c[idx])
        else:
            g = x
            h = y
        return g, h

class L2Torch():
    r"""L2 Norm using Torch and AD.

    Computes the :math:`\ell_2` norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2` using Torch and leveraging
    Automatic Differentiation for the gradient

    Parameters
    ----------
    b : :obj:`numpy.ndarray`
        Data vector
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    
    """

    def __init__(self, b, Op=None):
        self.Op = Op
        self.b = b

    def __call__(self, x, i):
        self.x = torch.from_numpy(x).requires_grad_()
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            f = (1. / 2.) * (torch.linalg.vector_norm(TorchOperator(Op).apply(self.x) -
                                                      torch.from_numpy(self.b[i])) ** 2)
        else:
            f = (1. / 2.) * (torch.linalg.vector_norm(self.x - torch.from_numpy(self.b[i])) ** 2)
        self.f = f
        return f.item()

    def grad(self, x, i):
        self.f.backward()
        g = self.x.grad.detach().numpy()
        return g
