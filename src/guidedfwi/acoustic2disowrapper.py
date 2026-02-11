__all__ = ["AcousticWave2D"]

from typing import Any, Optional, NewType, Type, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

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

class AcousticWave2D(NonlinearOperator):
    """Devito Acoustic propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Acoustic propagator

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple`
        Model origin in km ``(ox, oz)``
    spacing : :obj:`tuple`
        Model spacing in km ``(dx, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in km
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in km
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in km
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
    sub_gradient : :obj:'bool', optional
        If True, restricts the computation domain for each shot gather to a specified portion of the model.
        By default, the domain spans the maximum offset of the data plus an additional 1 km on 
        both the left and right sides.
    extent : :obj:`tuple`
        A tuple specifying the extent (in km) to extend the computation domain on the left and right for 
        sub_gradient. Default is (1.0, 1.0) km. This parameter is only used when sub_gradient is True.
    multisource_batch_size : int, optional
          Specifies the number of sources to be simulated simultaneously.
          Set to 1 for single-shot simulation or to the total number of sources
          to fire all sources at once. Intermediate values lead to random batching.
    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
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
        sub_gradient: Optional[bool] = False,
        extent: Optional[Tuple] = (1., 1.),
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
        self.vp_true = vp_true

        # Create vp if not provided and vprange is available
        if vprange is not None:
            vp = vprange[0] * np.ones(shape)
            vp[:, -1] = vprange[1]
        
        # Geometry parameters
        self.src = (src_x, src_z)
        self.rec = (rec_x, rec_z)

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
        self.sub_gradient = sub_gradient
        self.extent = extent
        self.multisource_batch_size = multisource_batch_size
        self.source_encoding = source_encoding
        self.encoding_params = encoding_params
        # Store encoding for reuse
        self.stored_encodings = None
        # Store model
        self.vp = vp

        # Inversion parameters
        self.loss = loss
        self.losshistory = []
        self.interm_gradients = []

        # MPI parameters
        self.base_comm = base_comm
    
        super().__init__(size=np.prod(shape), dtype=dtype)


    @staticmethod
    def _crop_model(m: NDArray, nbl: int, fs: bool) -> NDArray:
        """Remove absorbing boundaries from model"""
        if fs:
            return m[nbl:-nbl, :-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl]

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
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin in km ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing in km ``(dx, dz)``
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
        src_z: NDArray,
        rec_x: NDArray,
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
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in km
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in km
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
        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

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
                                         self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        return model, geometry

    def _get_location(self, isrc: int):
        # Calculate maximum offset in grid units
        max_offset = (self.rec[0][-1] - self.rec[0][0])

        # Compute x0 with grid conversion and boundary checking
        x0 = max(0, math.floor((self.src[0][isrc] - self.extent[0]) / self.spacing[0]))
        
        # Compute xf with grid conversion and boundary checking
        xf = min(math.ceil((self.src[0][isrc] + self.extent[1] + max_offset) / self.spacing[0]), self.shape[0] - 1)
    
        return (x0, xf)
    
    def recompute_encoding(self):
        """Recompute and store encoding for the next FWI iteration."""
        if self.source_encoding is not None:
            model = self._create_model(self.shape, self.origin, self.spacing,
                                    self.vp, self.space_order, self.nbl, self.fs)
            _, self.stored_encodings, _ = self._get_full_encoded_source(model)

    
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
                                         self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        
        # Update source location in geometry
        geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
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
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc])

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
        if not self.sub_gradient:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    self.vp, self.space_order, self.nbl, self.fs)

        # Run modelling
        nsrc = self.src[0].size
        dtot = []
        for isrc in range(nsrc):
            if self.sub_gradient:
                x0, xf = self._get_location(isrc)
    
                model = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                        self.vp[x0:xf], self.space_order, self.nbl, self.fs)
            d, dt = self._mod_oneshot(model, isrc, dt)
            if isrc == 0:
                nt_max = d.shape[0]
            elif d.shape[0] < nt_max:
                nt_max = d.shape[0]
            dtot.append(d)
            if self.clearcache:
                clear_devito_cache()
        dtot = np.array([d[:nt_max] for d in dtot]).reshape(nsrc, nt_max, d.shape[1])
        
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
                                            self.src[0], self.src[1],
                                            self.rec[0], self.rec[1],
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


    def mod_allshots_multisrc(self, dt=None, indices=None, vp_true=None, encodings=None) -> np.ndarray:
        """
        Forward model data using multi-source batching.

        For the batched case, this function:
        - Uses the provided vp_true (if available) to select the model.
        - Computes the full, encoded source for all shots using _get_full_encoded_source().
        - Generates (or uses provided) shot indices to randomize the order.
        - Extracts the pre-encoded source for each batch and uses it in the forward simulation.

        Parameters
        ----------
        dt : float, optional
            Time sampling.
        indices : array-like, optional
            Optional shot ordering. If None, a random ordering is generated.
        vp_true : np.ndarray, optional
            If provided, the true velocity model is used to model observed data.
        encodings : any, optional
            (Not used externally; encoding is generated internally.)

        Returns
        -------
        dtot : np.ndarray
            Modeled data arranged in batches of shape (num_batches, nt, nrec).
        indices : np.ndarray
            The shot ordering used.
        all_encoding : np.ndarray or None
            The full encoding (as generated by the Encoding class) if applicable.
        """
        nsrc = self.src[0].size

        # --- Case 1: Single-shot simulation ---
        if self.multisource_batch_size == 1:
            d, dt = self.mod_allshots(dt)
            return d

        # --- Select Model ---
        # Use vp_true if provided, else use self.vp.
        if vp_true is not None:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    vp_true, self.space_order, self.nbl, self.fs)
        else:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    self.vp, self.space_order, self.nbl, self.fs)

        # --- Build Full-Encoded Source for All Shots ---
        full_src, src_encodings, _ = self._get_full_encoded_source(model, encodings=encodings)
        if encodings is None:
            encodings = src_encodings
        # --- Generate Shot Ordering if Not Provided ---
        if indices is None:
            indices = np.arange(nsrc)
            np.random.shuffle(indices)

        batch_size = self.multisource_batch_size
        num_batches = nsrc // batch_size + (1 if nsrc % batch_size != 0 else 0)
        # if self.source_encoding == "random_time_delay":
        #     delay = self.encoding_params.get("delay", 0.2)
        #     geom_tn = self.tn * (1 + delay)
        # else:
        #     geom_tn = self.tn

        # --- Loop Over Batches ---
        for ii, start in enumerate(range(0, nsrc, batch_size)):
            # Extract the current batch of shot indices.
            batch_idx = indices[start:start + batch_size]

            batch_src_data = full_src[:, batch_idx]
            # Build geometry for this batch.
            
            geometry = self._create_geometry(model,
                                            self.src[0][batch_idx], self.src[1][batch_idx],
                                            self.rec[0], self.rec[1],
                                            self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
            # Build the source for this batch.
            if self.wav is None:
                src = geometry.src  # Expect src.data shape to be (nt, batch_size)
            else:
                src = CustomSource(name='src', grid=model.grid,
                                wav=self.wav, npoint=batch_src_data.shape[1],
                                time_range=geometry.time_axis)
                src.coordinates.data[:, :] = np.column_stack((self.src[0][batch_idx],
                                                            self.src[1][batch_idx]))
            # Replace the source data with the pre-encoded batch data.
            src.data[:, :] = batch_src_data

            # Create a solver for this batch and run forward simulation.
            solver = AcousticWaveSolver(model, geometry, space_order=self.space_order)
            d_batch, _, _, _ = solver.forward(vp=model.vp, src=src, autotune=True)
            if dt is None:
                d_batch = d_batch.data.copy()
            else:
                d_batch = d_batch.resample(dt).data.copy()

            dtot.append(d_batch)

        dtot = np.array(dtot).reshape(num_batches, d_batch.shape[0], d_batch.shape[1])
        return dtot, indices, encodings


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
        # Identify number of shots
        if isrcs is None:
            nsrc = self.src[0].size
            isrcs = range(nsrc)
        
        # Create model with class vp to define a geometry and time axis consistent with 
        # the observed data and one with provided vp (to be used as input for loss and
        # gradient computation)
        if not self.sub_gradient:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    self.vp, self.space_order, self.nbl, self.fs)
            modelvp = self._create_model(self.shape, self.origin, self.spacing, 
                                        vp, self.space_order, self.nbl, self.fs)
            
            

            # Geometry for single source
            geometry = self._create_geometry(model,
                                            self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
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
            if self.sub_gradient:
                x0, xf = self._get_location(isrc)

                model = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                        self.vp[x0:xf], self.space_order, self.nbl, self.fs)
                
                modelvp = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                            vp[x0:xf], self.space_order, self.nbl, self.fs)
                geometry = self._create_geometry(model,
                                         self.src[0][isrc:isrc+1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
                # Re-create source (if wav is not None)
                if self.wav is None:
                    src = geometry.src
                else:
                    src = CustomSource(name='src', grid=model.grid,
                                    wav=self.wav, npoint=1,
                                    time_range=geometry.time_axis)
                solver = AcousticWaveSolver(model, geometry,
                                    space_order=self.space_order)
            # Update source location in geometry
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc])
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
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad[1].data[:], self.nbl, self.fs)
                        full_grad = np.zeros(self.shape, dtype=np.float64)
                        full_grad[x0:xf] = grad.copy()
                    else:
                        grad = lossgrad[1].data[:]
                else:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad[1].data[:], self.nbl, self.fs)
                        full_grad[x0:xf] += grad.copy()
                    else:
                        grad += lossgrad[1].data[:]
            elif computeloss:
                loss += lossgrad
            elif computegrad:
                if isrc == 0:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad.data[:], self.nbl, self.fs)
                        full_grad = np.zeros(self.shape, dtype=np.float64)
                        full_grad[x0:xf] = grad.copy()
                    else:
                        grad = lossgrad.data[:]
                else:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad.data[:], self.nbl, self.fs)
                        full_grad[x0:xf] += grad.copy()
                    else:
                        grad += lossgrad.data[:]
            
        if self.clearcache:
                clear_devito_cache()

        # Gather gradients
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad = self.base_comm.allreduce(grad, op=MPI.SUM)

        # Postprocess loss and gradient
        
        grad = self._crop_model(grad, self.nbl, self.fs) if not self.sub_gradient else full_grad
        if self.sub_gradient:
            modelvp_ = self._create_model(self.shape, self.origin, self.spacing, 
                                        vp, self.space_order, self.nbl, self.fs)
            vp = self._crop_model(modelvp_.vp.data[:], self.nbl, self.fs)
        else:
            vp = self._crop_model(modelvp.vp.data[:], self.nbl, self.fs)
        if postprocess is not None:
            loss, grad = postprocess(vp, loss, grad)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad
    
    def reset_encoding(self):
        self.stored_encodings = None
        self.stored_indices = None

    def _multi_source_loss_grad(self, vp, postprocess=None, computeloss=True, computegrad=True, 
                               unstacked_grad=False, denoiser=None, mask=None):
        """
        """
        vp = vp.reshape(self.shape)
        nsrc = self.src[0].size
        acc_loss = 0
        acc_loss_i = 0.0
        full_grad = np.zeros(self.shape, dtype=np.float64)
        # Create the synthetic model using the current inversion model vp.
        model = self._create_model(self.shape, self.origin, self.spacing,
                                     self.vp, self.space_order, self.nbl, self.fs)
        modelvp = self._create_model(self.shape, self.origin, self.spacing,
                                     vp, self.space_order, self.nbl, self.fs)
        

        # Use the **stored** encoding to keep it fixed across L-BFGS iterations
        if self.stored_encodings is None:
            full_src, self.stored_encodings, enc_obj = self._get_full_encoded_source(model)  # Compute if missing
            self.stored_indices = np.arange(nsrc)
            if self.multisource_batch_size > 1:
                np.random.shuffle(self.stored_indices)
        else:
            full_src = self._get_full_encoded_source(model, encodings=self.stored_encodings)[0]  # Keep encoding fixed
            enc_obj = Encoding(self.source_encoding, nsrc, self.encoding_params)
            enc_obj.enc = self.stored_encodings  # <-- force same encoding object

        
        self.loss.encoder = enc_obj
        self.loss.encode_data()

        indices = self.stored_indices
        batch_size = self.multisource_batch_size
        num_batches = nsrc // batch_size + (1 if nsrc % batch_size != 0 else 0)
  
        partial_grads = [] if unstacked_grad else None
        batch_srcs = [] if unstacked_grad else None
        
        # Loop Over Batches
        for ii, start in enumerate(range(0, nsrc, batch_size)):
            # Extract the current batch of shot indices.
            batch_idx = indices[start:start + batch_size]

            batch_src_data = full_src[:, batch_idx]
            # Build geometry for this batch.
            
            geometry = self._create_geometry(model,
                                            self.src[0][batch_idx], self.src[1][batch_idx],
                                            self.rec[0], self.rec[1],
                                            self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
            # Build the source for this batch.
            if self.wav is None:
                src = geometry.src  # Expect src.data shape to be (nt, batch_size)
            else:
                src = CustomSource(name='src', grid=model.grid,
                                wav=self.wav, npoint=batch_src_data.shape[1],
                                time_range=geometry.time_axis)
                src.coordinates.data[:, :] = np.column_stack((self.src[0][batch_idx],
                                                            self.src[1][batch_idx]))
            # Replace the source data with the pre-encoded batch data.
            src.data[:, :] = batch_src_data
            # print(src.data[:, 0].max())
            # Create a solver for this batch.
            solver = AcousticWaveSolver(model, geometry, space_order=self.space_order)
            # Run forward simulation for the batch.
            adjsrc, u0, usnaps, _ = solver.forward(vp=modelvp.vp, src=src, autotune=True, save=geometry.nt,
                                                   factor=self.factor)

            # Compute loss
            if computeloss:
                loss_i = self.loss(adjsrc.data[:].ravel(), batch_idx)
                # acc_loss += loss_i
            if computegrad:
                # Compute adjoint source
                adjsrc.data[:] = self.loss.grad(adjsrc.data[:].ravel(), batch_idx).reshape(adjsrc.data.shape)
                # Compute gradient
                grad_i, _ = solver.gradient(rec=adjsrc, u=u0, usnaps=usnaps, vp=modelvp.vp, checkpointing=self.checkpointing, autotune=True,
                                        factor=self.factor)
                grad_i = self._crop_model(grad_i.data[:], self.nbl, self.fs)
                if postprocess is not None:
                        loss_i, grad_i = postprocess(vp, loss_i, grad_i)
                full_grad += grad_i
                acc_loss += loss_i
                if unstacked_grad:
                    partial_grads.append(grad_i)
                    batch_srcs.append(batch_idx)
                    
        # Gather gradients
        if self.base_comm is not None:
            if computeloss:
                acc_loss = self.base_comm.allreduce(acc_loss, op=MPI.SUM)
            if computegrad:
                full_grad = self.base_comm.allreduce(full_grad, op=MPI.SUM)

        # if postprocess is not None:
        #     acc_loss, full_grad = postprocess(vp, acc_loss, full_grad)

        partial_grads = np.array(partial_grads, dtype=np.float64) if unstacked_grad else None
        if denoiser is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            denoiser = denoiser.to(device)
            denoiser.eval()
            # Normalize each partial gradient
            # if partial_grads.shape[0] > 1:
            #     for pg in range(partial_grads.shape[0]):
            gmax = np.abs(partial_grads).max()
            if gmax > 0:
                partial_grads = partial_grads / gmax
  
            # Convert grad (numpy) to tensor, add batch and channel dimensions, and move to device.
            grad_tensor = torch.from_numpy(partial_grads).unsqueeze(0).float().to(device)
            
            # Denoise and convert back to numpy.
            grad = denoiser(grad_tensor) * gmax
            full_grad = grad.squeeze().detach().cpu().numpy()
            if mask is not None:
                full_grad = full_grad * mask

        batch_srcs = np.array(batch_srcs, dtype=int) if unstacked_grad else None
        
        # Save loss history
        if computeloss:
            self.losshistory.append(acc_loss)

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
    
    def gen_multi_source_grad(self, vp, postprocess=None, computeloss=True, computegrad=True):
        """
        """
        vp = vp.reshape(self.shape)
        nsrc = self.src[0].size
        gradients = []
        stored_wavefields = {}
        full_grad = np.zeros(self.shape, dtype=np.float64)
        # Create the synthetic model using the current inversion model vp.
        model = self._create_model(self.shape, self.origin, self.spacing,
                                     self.vp, self.space_order, self.nbl, self.fs)
        modelvp = self._create_model(self.shape, self.origin, self.spacing,
                                     vp, self.space_order, self.nbl, self.fs)
        
        # Compute and store wavefields for each shot **only once**
        for isrc in range(nsrc):
            # Create geometry for each shot
            geometry = self._create_geometry(model, [self.src[0][isrc]], [self.src[1][isrc]],
                                            self.rec[0], self.rec[1], self.t0, self.tn,
                                            self.src_type, f0=self.f0, dt=self.dt)

            # Run forward modeling **once per shot**
            solver = AcousticWaveSolver(model, geometry, space_order=self.space_order)
            _, u0, _, _ = solver.forward(vp=modelvp.vp, save=True)

            # Store wavefield for this shot
            stored_wavefields[isrc] = u0
        
        # Generate a new encoding
        enc_obj = Encoding(self.source_encoding, nsrc, self.encoding_params)

        # Use the **stored** encoding to keep it fixed across L-BFGS iterations
        if self.stored_encodings is None:
            full_src, self.stored_encodings, enc_obj = self._get_full_encoded_source(model)  # Compute if missing
        else:
            full_src = self._get_full_encoded_source(model, encodings=self.stored_encodings)[0]  # Keep encoding fixed
            enc_obj = None
        if enc_obj is not None:
            self.loss.encoder = enc_obj
            self.loss.encode_data()

        indices = np.arange(nsrc)
        np.random.shuffle(indices)

        batch_size = self.multisource_batch_size
        num_batches = nsrc // batch_size + (1 if nsrc % batch_size != 0 else 0)
  

        # --- Loop Over Batches ---
        for ii, start in enumerate(range(0, nsrc, batch_size)):
            # Extract the current batch of shot indices.
            batch_idx = indices[start:start + batch_size]

            batch_src_data = full_src[:, batch_idx]
            # Build geometry for this batch.
            
            geometry = self._create_geometry(model,
                                            self.src[0][batch_idx], self.src[1][batch_idx],
                                            self.rec[0], self.rec[1],
                                            self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
            # Build the source for this batch.
            if self.wav is None:
                src = geometry.src  # Expect src.data shape to be (nt, batch_size)
            else:
                src = CustomSource(name='src', grid=model.grid,
                                wav=self.wav, npoint=batch_src_data.shape[1],
                                time_range=geometry.time_axis)
                src.coordinates.data[:, :] = np.column_stack((self.src[0][batch_idx],
                                                            self.src[1][batch_idx]))
            # Replace the source data with the pre-encoded batch data.
            src.data[:, :] = batch_src_data
            # Create a solver for this batch.
            solver = AcousticWaveSolver(model, geometry, space_order=self.space_order)
            # Run forward simulation for the batch.
            adjsrc, u0, usnaps, _ = solver.forward(vp=modelvp.vp, src=src, autotune=True, save=geometry.nt,
                                                   factor=self.factor)

            # Compute loss
            if computeloss:
                acc_loss += self.loss.loss(adjsrc.data[:].ravel(), batch_idx)
            if computegrad:
                # Compute adjoint source
                adjsrc.data[:] = self.loss.grad(adjsrc.data[:].ravel(), batch_idx).reshape(adjsrc.data.shape)
                # Compute gradient
                grad, _ = solver.gradient(rec=adjsrc, u=u0, usnaps=usnaps, vp=modelvp.vp, checkpointing=self.checkpointing, autotune=True,
                                        factor=self.factor)
                full_grad += self._crop_model(grad.data[:], self.nbl, self.fs)
        
        if postprocess is not None:
            acc_loss, full_grad = postprocess(vp, acc_loss, full_grad)

        # Save loss history
        if computeloss:
            self.losshistory.append(acc_loss)

        if computeloss and computegrad:
            return acc_loss, full_grad.ravel()
        elif computeloss:
            return acc_loss
        else:
            return full_grad.ravel()
        
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

        # Dispatch based on multisource_batch_size
        if self.multisource_batch_size == 1:
            # Single-shot simulation (existing single-shot routine)
            lossgrad = self._loss_grad(vp.reshape(self.shape),
                                       postprocess=postprocess,
                                       computeloss=computeloss,
                                       computegrad=computegrad)
        else:
            # Batched multi-source simulation
            lossgrad = self._multi_source_loss_grad(vp.reshape(self.shape),
                                                   postprocess=postprocess,
                                                   computeloss=computeloss,
                                                   computegrad=computegrad)

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