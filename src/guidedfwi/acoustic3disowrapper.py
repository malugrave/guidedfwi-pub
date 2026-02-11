__all__ = ["AcousticWave3D"]

from typing import Any, Optional, NewType, Type, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.model import SeismicModel

from .acousticisosolver import AcousticWaveSolver
from .source import CustomSource
from .utils import clear_devito_cache
from .nonlinear import NonlinearOperator
from .encoding import Encoding
from .loss import L2MultiSource

try:
    from mpi4py import MPI
    mpitype = MPI.Comm
except:
    mpitype = Any

MPIType = NewType("MPIType", mpitype)

class AcousticWave3D(NonlinearOperator):
    """Devito Acoustic propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Acoustic propagator

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, ny, nz)``
    origin : :obj:`tuple`
        Model origin ``(ox, oy, oz)``
    spacing : :obj:`tuple`
        Model spacing ``(dx, dy, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in km
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in km
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in km
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in km
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in km
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in km
    t0 : :obj:`float`
        Initial time in s
    tn : :obj:`float`
        Final time in s
    dt : :obj:`float`, optional
        Time step in s (if not provided this is directly inferred by devito)
    vp : :obj:`numpy.ndarray`, optional
        Velocity model in km/s for modelling of size :math:`n_x \times n_z`
        (use ``None`` if the data is already available)
    vprange : :obj:`tuple`, optional
        Velocity range in km/s ``(vmin, vmax)``, to be used in loss and gradient computations
        (can be provided instead of ``vp`` to create a propagator with a time axis 
        that is consistent with that of the data modelled with ``vp``)
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    src_type : :obj:`str`, optional
        Source type
    f0 : :obj:`float`, optional
        Source peak frequency in Hz
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` will be ignored)
    fs : :obj:'bool', optional
        Use free surface boundary at the top of the model.
    streamer_acquisition : :obj:'bool', optional
        Update receiver locations in geometry for each source
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models.
        Cannot be used with snapshotting (factor).
    factor : :obj:`int`, optional 
        Subsampling factor to use snapshots of the wavefield to compute the gradient.
        Cannot be used with checkpointing.
    loss : :obj:`Type`, optional
        Loss object.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    clearcache : :obj:`bool`, optional
        Clear devito cache (``True``) or not (``False``) after every modelling step
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    multisource_batch_size : :obj:`int`, optional
        Number of simultaneous sources per batch. Defaults to ``1`` (single‑source).
    source_encoding : :obj:`str`, optional
        Encoding scheme to use for multi‑source acquisition (e.g. “Hadamard”, “Gaussian”).
    encoding_params : :obj:`dict`, optional
        Parameters for the chosen source encoding (e.g. seed, noise level).
    vp_true : :obj:`numpy.ndarray`, optional
        True velocity model used as reference for encoding or regularization.    
    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        dt: Optional[float] = None,
        vp: Optional[NDArray] = None,
        vprange: Optional[Tuple] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        fs: Optional[bool] = False,
        streamer_acquisition: Optional[bool] = False,
        checkpointing: Optional[bool] = False,
        factor: Optional[int] = None,
        loss: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
        clearcache: Optional[bool] = False,
        base_comm: Optional[MPIType] = None,
        multisource_batch_size: int = 1,
        source_encoding: Optional[str] = None,
        encoding_params: Optional[dict] = None,
        vp_true: Optional[NDArray] = None,
    ) -> None:

        # Check to ensure that vp or vprange is provided
        if vp is None and vprange is None:
            raise ValueError("Provide either vp or vprange, not none...")
        elif vp is not None and vprange is not None:
            raise ValueError("Provide either vp or vprange, not both...")

        # Create vp if not provided and vprange is available
        if vprange is not None:
            vp = vprange[0] * np.ones(shape)
            vp[-1, -1] = vprange[1]
        
        # Geometry parameters
        self.src = (src_x, src_y, src_z)
        self.rec = (rec_x, rec_y, rec_z)
        
        # # Gather full shot list across all ranks
        # if (base_comm is not None) and (multisource_batch_size > 1):
        #     all_x = base_comm.allgather(src_x)
        #     all_y = base_comm.allgather(src_y)
        #     all_z = base_comm.allgather(src_z)
            
        #     # Flatten into global arrays
        #     self.global_src_x = np.concatenate(all_x)
        #     self.global_src_y = np.concatenate(all_y)
        #     self.global_src_z = np.concatenate(all_z)
            
        #     # Override for multi‐source
        #     self.src = (self.global_src_x, self.global_src_y, self.global_src_z)

        # Modelling parameters
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
        self.t0 = t0
        self.tn = tn
        self.dt = dt
        self.space_order = space_order
        self.nbl = nbl
        self.src_type = src_type
        self.f0 = f0
        self.wav = wav
        self.fs = fs
        self.streamer_acquisition = streamer_acquisition
        self.checkpointing = checkpointing
        self.factor = factor
        self.clearcache = clearcache
        
        # Store model
        self.vp = vp

        # Inversion parameters
        self.loss = loss
        self.losshistory = []

        # MPI parameters
        self.base_comm = base_comm
        
        # Multi‐source parameters (copied from 2D wrapper)
        self.multisource_batch_size = multisource_batch_size
        self.source_encoding = source_encoding
        self.encoding_params = encoding_params
        self.stored_encodings = None
        self.vp_true = vp_true
    
        super().__init__(size=np.prod(shape), dtype=dtype)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int, fs: bool) -> NDArray:
        """Remove absorbing boundaries from model"""
        if fs:
            return m[nbl:-nbl, nbl:-nbl, :-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 4,
        nbl: int = 20,
        fs: bool = False,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, ny, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oy, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dy, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries
        fs : :obj:'bool', optional
            Use free surface boundary at the top of the model.

        Returns
        -------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        
        """
        model = Model(
            space_order=space_order,
            vp=vp,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            fs=fs,
        )
        return model

    def _create_geometry(
        self,
        model,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str,
        f0: float = 20.0,
        dt: float = None
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in km
        src_y : :obj:`numpy.ndarray`
            Source y-coordinates in km
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in km
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in km
        rec_y : :obj:`numpy.ndarray`
            Receiver y-coordinates in km
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in km
        t0 : :obj:`float`
            Initial time in s
        tn : :obj:`float`
            Final time in s
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency in Hz
        dt : :obj:`float`, optional
            Time step time in s (if provided, the geometry time_axis is
            recreated with this time step)

        """
        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_y
        src_coordinates[:, 2] = src_z

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_y
        rec_coordinates[:, 2] = rec_z

        geometry = AcquisitionGeometry(
            model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0,
            fs=self.fs,
        )

        # Resample geometry to user defined dt
        if dt is not None:
            geometry.resample(dt)

        return geometry

    def model_and_geometry(self):
        model = self._create_model(self.shape, self.origin, self.spacing, 
                                   self.vp, self.space_order, self.nbl, self.fs)
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.src[2][:1], 
                                         self.rec[0], self.rec[1], self.rec[2], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        return model, geometry

    def _mod_oneshot(self, model: SeismicModel, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        isrc : :obj:`int`
            Index of source to model
        dt : :obj:`float`, optional
            Time sampling in s used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data of size ``nr \times nt``
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        # Create geometry
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.src[2][:1], 
                                         self.rec[0], self.rec[1], self.rec[2], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        
        # Update source location in geometry
        geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
        if self.streamer_acquisition:
            # Update receiver locations in geometry
            geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + self.rec[0]
        
        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])

        # Solve
        solver = AcousticWaveSolver(model, geometry, 
                                    space_order=self.space_order)
        d, _, _, _ = solver.forward(vp=model.vp, src=src, autotune=True)

        # Resample
        if dt is None:
            dt = geometry.dt
            d = d.data.copy()
        else:
            d = d.resample(dt).data.copy()
        
        return d, dt

    def mod_allshots(self, dt=None) -> NDArray:
        """FD modelling for all shots

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        # Create model
        model = self._create_model(self.shape, self.origin, self.spacing, 
                                   self.vp, self.space_order, self.nbl, self.fs)

        # Run modelling
        nsrc = self.src[0].size
        dtot = []
        for isrc in range(nsrc):
            d, dt = self._mod_oneshot(model, isrc, dt)
            dtot.append(d)
            if self.clearcache:
                clear_devito_cache()
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        
        return dtot, dt

    def mod_allshots_mpi(self, dt=None) -> NDArray:
        """FD modelling for all shots with mpi gathering

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s

        Returns
        -------
        d : :obj:`np.ndarray`
            Data for all shots
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        dtotrank, dt = self.mod_allshots(dt)

        # gather shots from all ranks
        dtot = np.concatenate(self.base_comm.allgather(dtotrank), axis=0)
        
        return dtot, dt

    def _adjoint_source(self, d_syn, isrc):
        """Adjoint source computation

        Note to self, takes flatten inputs and returns flatten outputs
        """
        return self.loss.grad(d_syn, isrc)

    def _loss_grad_oneshot(self, vp, src, solver, isrc,
                           computeloss=True, computegrad=True) -> Tuple[float, NDArray]:
        """Raw loss function and gradient for one shot

        Compute raw loss function and gradient for one shot without applying any pre/post-processing. Note
        that Devito returns the gradient for slowness square.

        """
        # Compute synthetic data and full forward wavefield u0
        adjsrc, u0, usnaps, _ = solver.forward(vp=vp, save=True if self.factor is None else False,
                                               src=src, autotune=True, factor=self.factor)
        
        # Compute loss
        if computeloss:
            loss = self.loss(adjsrc.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            adjsrc.data[:] = self._adjoint_source(adjsrc.data[:].ravel(), isrc).reshape(adjsrc.data.shape)

            # Compute gradient
            grad, _ = solver.gradient(rec=adjsrc, u=u0, usnaps=usnaps, vp=vp, checkpointing=self.checkpointing, autotune=True,
                                      factor=self.factor)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def _loss_grad(self, vp, isrcs=None, postprocess=None, computeloss=True, computegrad=True):
        """Compute loss function and gradient
        
        Parameters
        ----------
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s
        isrcs : :obj:`list`, optional
            Indices of shots to be used in gradient computation 
            (if ``None``, use all shots whose number is inferred from ``dobs``)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """
        # Create model with class vp to define a geometry and time axis consistent with 
        # the observed data and one with provided vp (to be used as input for loss and
        # gradient computation)
        model = self._create_model(self.shape, self.origin, self.spacing, 
                                   self.vp, self.space_order, self.nbl, self.fs)
        modelvp = self._create_model(self.shape, self.origin, self.spacing, 
                                     vp, self.space_order, self.nbl, self.fs)
        
        # Identify number of shots
        if isrcs is None:
            nsrc = self.src[0].size
            isrcs = range(nsrc)

        # Geometry for single source
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.src[2][:1], 
                                         self.rec[0], self.rec[1], self.rec[2], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)

        # Solver
        solver = AcousticWaveSolver(model, geometry,
                                    space_order=self.space_order)
        
        # Compute loss and gradient
        loss = 0.
        for isrc in isrcs:
            # Update source location in geometry
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            if self.streamer_acquisition:
                # Update receiver locations in geometry
                geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + self.rec[0]
            
            # Compute loss and gradient for one shot
            lossgrad = self._loss_grad_oneshot(modelvp.vp, src, solver, isrc,
                                               computeloss=computeloss, 
                                               computegrad=computegrad)
            if computeloss and computegrad:
                loss += lossgrad[0]
                if isrc == 0:
                    grad = lossgrad[1].data[:]
                else:
                    grad += lossgrad[1].data[:]
            elif computeloss:
                loss += lossgrad
            elif computegrad:
                if isrc == 0:
                    grad = lossgrad.data[:]
                else:
                    grad += lossgrad[1].data[:]
            
        if self.clearcache:
                clear_devito_cache()

        # Gather gradients
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad = self.base_comm.allreduce(grad, op=MPI.SUM)

        # Postprocess loss and gradient
        grad = self._crop_model(grad, self.nbl, self.fs)
        vp = self._crop_model(modelvp.vp.data[:], self.nbl, self.fs)
        if postprocess is not None:
            loss, grad = postprocess(vp, loss, grad)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def loss_grad(self, x, convertvp=None, postprocess=None,
                  computeloss=True, computegrad=True,
                  debug=False, gradlims=None):
        """Compute loss function and gradient to be used by solver

        This routine wraps _loss_grad providing and returning numpy arrays 
        and should be used with any solver
        
        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient
        debug : :obj:`bool`, optional
            Debugging flag
        gradlims : :obj:`tuple`, optional
            Limits of gradient to be used in plotting when ``debug=True``

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """

        # Convert x to velocity
        if convertvp is None:
            vp = x.reshape(self.shape)
        else:
            vp = convertvp(x.reshape(self.shape))

        # Dispatch between single‐shot and batched multi‐source
        if self.multisource_batch_size == 1:
            lossgrad = self._loss_grad(
                vp.reshape(self.shape),
                postprocess=postprocess,
                computeloss=computeloss,
                computegrad=computegrad
            )
        else:
            lossgrad = self._multi_source_loss_grad(
                vp.reshape(self.shape),
                postprocess=postprocess,
                computeloss=computeloss,
                computegrad=computegrad
            )

        # Split lossgrad based on what has been computed in self._loss_grad
        if computeloss and computegrad:
            loss, grad = lossgrad
        elif computeloss:
            loss, grad = lossgrad, None
        else:
            loss, grad = None, lossgrad

        # Save loss history
        if computeloss:
            self.losshistory.append(loss)
        
        # Display results in debugging mode
        if debug and computeloss and computegrad:
            print('Debug - loss, grad.min(), grad.max()',
                  loss, grad.min(), grad.max())
            plt.figure()
            plt.imshow(grad.T, vmin=gradlims[0] if gradlims is not None else -grad.max(),
                       vmax=gradlims[1] if gradlims is not None else grad.max(),
                       aspect='auto', cmap='seismic')
            plt.colorbar()

        # Return loss, grad or both
        if computeloss and computegrad:
            return loss, grad.ravel()
        elif computeloss:
            return loss
        else:
            return grad.ravel()

    def loss(self, x, convertvp=None, postprocess=None):
        """Compute loss function to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        loss : :obj:`float`
            Loss function

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=True, computegrad=False)

    def grad(self, x, convertvp=None, postprocess=None):
        """Compute gradient to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=False, computegrad=True)

    def _get_full_encoded_source(self, model, encodings=None):
        """
        Build a geometry covering all shots and extract the full source data.
        Then apply source encoding (if enabled) using the Encoding class.
        
        Returns
        -------
        full_src : np.ndarray
            Encoded source data of shape (nt, nsrc).
        geometry_all : object
            The geometry built over all shots.
        """
        # Extend tn if using random time delay.
        # if self.source_encoding == "random_time_delay":
        #     delay = self.encoding_params.get("delay", 0.2)
        #     geom_tn = self.tn * (1 + delay)
        # else:
        #     geom_tn = self.tn
        # Create a geometry that spans all shots.
        geometry_all = self._create_geometry(model,
                                            self.src[0], self.src[1], self.src[2],
                                            self.rec[0], self.rec[1], self.rec[2],
                                            self.t0, self.tn, 
                                            self.src_type, f0=self.f0, dt=self.dt)
        # Extract full-source data.
        if self.wav is None:
            # Assume geometry_all.src.data has shape (nt, nsrc)
            full_src = geometry_all.src.data.copy()
        else:
            # Build a CustomSource for all shots.
            src_all = CustomSource(name='src_all', grid=model.grid,
                                wav=self.wav, npoint=self.src[0].size,
                                time_range=geometry_all.time_axis)
            src_all.coordinates.data[:, :] = np.column_stack((self.src[0], self.src[1]))
            full_src = src_all.data.copy()  # Expected shape: (nt, nsrc)
        full_src = np.array(full_src.T)
        # print(full_src.shape)
        # Apply encoding if requested.
        if self.source_encoding in ['random_polarity', 'random_time_delay']:
            enc_obj = Encoding(self.source_encoding, self.src[0].size, self.encoding_params)
            if encodings is not None:
                enc_obj.enc = encodings
            encoded_full_src = enc_obj.apply(full_src)
            encodings = enc_obj.enc
        else:
            encoded_full_src = full_src
            encodings = None
            enc_obj = None

        return encoded_full_src.T, encodings, enc_obj
        
    def _multi_source_loss_grad(
        self,
        vp,
        postprocess=None,
        computeloss=True,
        computegrad=True,
        unstacked_grad=False,
        denoiser=None,
        mask=None,
    ):
        """
        Batched multi‑source loss & gradient (3D).

        Parameters
        ----------
        vp : :obj:`numpy.ndarray`
            3D velocity model array of shape ``(nx, ny, nz)``.
        postprocess : callable, optional
            If provided, a function ``(vp, loss, grad) -> (loss, grad)``.
        computeloss : :obj:`bool`, optional
            Whether to compute and accumulate the loss. Default is ``True``.
        computegrad : :obj:`bool`, optional
            Whether to compute and accumulate the gradient. Default is ``True``.
        unstacked_grad : :obj:`bool`, optional
            If ``True``, returns per‑batch gradients and source indices.
        denoiser : callable, optional
            If provided, a PyTorch module to denoise per‑batch grads.
        mask : :obj:`numpy.ndarray`, optional
            Mask to apply to the final gradient after denoising.

        Returns
        -------
        If `computeloss` and `computegrad`:
            Tuple[`float`, `numpy.ndarray`]
            Total loss and flattened gradient.
        If `computeloss` only:
            `float`
            Total loss.
        If `computegrad` only:
            `numpy.ndarray`
            Flattened gradient.
        If `unstacked_grad`:
            Tuple[List[`numpy.ndarray`], `numpy.ndarray`]
            List of per‑batch gradients and array of batch source indices.
        """
        
        # Reshape & prep
        vp = vp.reshape(self.shape)
        nsrc = self.src[0].size
        acc_loss = 0.0
        full_grad = np.zeros(self.shape, dtype=np.float32)

        # Build models
        model   = self._create_model(self.shape, self.origin, self.spacing,
                                     self.vp,   self.space_order, self.nbl, self.fs)
        modelvp = self._create_model(self.shape, self.origin, self.spacing,
                                     vp,       self.space_order, self.nbl, self.fs)

        # Get—or reuse—encoded full source over all shots
        if self.stored_encodings is None:
            full_src, self.stored_encodings, enc_obj = \
                self._get_full_encoded_source(model)
            self.stored_indices = np.arange(nsrc)
            if self.multisource_batch_size > 1:
                np.random.shuffle(self.stored_indices)
        else:
            full_src = self._get_full_encoded_source(
                model, encodings=self.stored_encodings
            )[0]
            enc_obj = Encoding(self.source_encoding,
                               nsrc,
                               self.encoding_params)
            enc_obj.enc = self.stored_encodings

        # Source batching setup
        if self.multisource_batch_size > 1:
            indices = self.stored_indices
            
            # # Optionally partition sources across ranks for true parallelism:
            # world_size = comm.Get_size()
            # rank       = comm.Get_rank()
            # all_idx    = self.stored_indices
            # indices    = all_idx[rank::world_size]
        else:
            indices = np.arange(nsrc)
        batch_size = self.multisource_batch_size

        partial_grads = [] if unstacked_grad else None
        batch_srcs   = [] if unstacked_grad else None

        # Do loop over source batches
        for start in range(0, nsrc, batch_size):
            batch_idx     = indices[start : start + batch_size]
            batch_src_data = full_src[:, batch_idx]

            # Build 3D geometry
            geometry = self._create_geometry(
                model,
                self.src[0][batch_idx],
                self.src[1][batch_idx],
                self.src[2][batch_idx],
                self.rec[0],
                self.rec[1],
                self.rec[2],
                self.t0,
                self.tn,
                self.src_type,
                f0=self.f0,
                dt=self.dt,
            )

            # Assemble CustomSource if needed
            if self.wav is None:
                src = geometry.src
            else:
                src = CustomSource(
                    name="src",
                    grid=model.grid,
                    wav=self.wav,
                    npoint=batch_idx.size,
                    time_range=geometry.time_axis,
                )
                # set each (x,y,z)
                src.coordinates.data[:, :] = np.column_stack((
                    self.src[0][batch_idx],
                    self.src[1][batch_idx],
                    self.src[2][batch_idx],
                ))

            # Overwrite with encoded data
            src.data[:, :] = batch_src_data

            # Compute forward / adjoint
            solver = AcousticWaveSolver(
                model, geometry, space_order=self.space_order
            )
            adjsrc, u0, usnaps, _ = solver.forward(
                vp=modelvp.vp,
                src=src,
                autotune=True,
                save=True if self.factor is None else False,
                factor=self.factor,
            )

            if computeloss:
                loss_i = self.loss(adjsrc.data[:].ravel(), batch_idx)

            if computegrad:
                adjsrc.data[:] = self.loss.grad(
                    adjsrc.data[:].ravel(), batch_idx
                ).reshape(adjsrc.data.shape)

                grad_i, _ = solver.gradient(
                    rec=adjsrc,
                    u=u0,
                    usnaps=usnaps,
                    vp=modelvp.vp,
                    checkpointing=self.checkpointing,
                    autotune=True,
                    factor=self.factor,
                )
                # crop away absorbing boundaries
                grad_i = self._crop_model(grad_i.data[:], self.nbl, self.fs)

                if postprocess is not None:
                    loss_i, grad_i = postprocess(vp, loss_i, grad_i)

            full_grad += grad_i
            
            if computeloss:
                acc_loss += loss_i
                
            if unstacked_grad:
                partial_grads.append(grad_i)
                batch_srcs.append(batch_idx)

        # MPI reduction
        # if self.base_comm is not None:
        #     if computeloss:
        #         acc_loss = self.base_comm.allreduce(acc_loss, op=MPI.SUM)
        #     if computegrad:
        #         full_grad = self.base_comm.allreduce(full_grad, op=MPI.SUM)

        if self.base_comm is not None:
            world_size = self.base_comm.Get_size()

            if computeloss:
                # sum then average so loss is invariant to number of ranks
                acc_loss = self.base_comm.allreduce(acc_loss, op=MPI.SUM) / world_size

            if computegrad:
                # sum then average so gradient magnitude is constant
                full_grad = self.base_comm.allreduce(full_grad, op=MPI.SUM) / world_size

        # Optional denoiser on per‐batch grads
        if denoiser is not None:
            gmax = np.max(np.abs(partial_grads))
            if gmax > 0:
                partial_grads = [g / gmax for g in partial_grads]
            device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grad_tensor = torch.from_numpy(
                np.stack(partial_grads)
            ).unsqueeze(0).float().to(device)
            grad = denoiser(grad_tensor) * gmax
            full_grad = grad.squeeze().detach().cpu().numpy()
            if mask is not None:
                full_grad *= mask

        # Save & return
        if computeloss:
            self.losshistory.append(acc_loss)

        batch_srcs = np.array(batch_srcs, dtype=int) if unstacked_grad else None

        if computeloss and computegrad:
            if unstacked_grad:
                return acc_loss, full_grad.ravel(), partial_grads, batch_srcs
            return acc_loss, full_grad.ravel()
        elif computeloss:
            return acc_loss
        else:
            if unstacked_grad:
                return partial_grads, batch_srcs
            return full_grad.ravel()