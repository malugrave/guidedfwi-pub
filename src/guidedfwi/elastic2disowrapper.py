__all__ = ["ElasticWave2D"]

from typing import Any, Optional, NewType, Type, Tuple

import numpy as np

from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from devito import Function
from examples.seismic import AcquisitionGeometry, Receiver

from .source import CustomSource
from .elasticisosolver import ElasticWaveSolver, ElasticSeismicModel
from .nonlinear import NonlinearOperator
from .utils import clear_devito_cache

try:
    from mpi4py import MPI
    mpitype = MPI.Comm
except:
    mpitype = Any

MPIType = NewType("MPIType", mpitype)


class ElasticWave2D():
    """Devito Elastic propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Elastic propagator

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple`
        Model spacing ``(dx, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time in ms
    tn : :obj:`int`
        Final time in ms
    dt : :obj:`float`, optional
        Time step in ms (if not provided this is directly inferred by devito)
    m1 : :obj:`numpy.ndarray`, optional
        First elastic modulus, 
        (use ``None`` if the data is already available)
    m2 : :obj:`numpy.ndarray`, optional
        Second elastic modulus, 
        (use ``None`` if the data is already available)
    m3 : :obj:`numpy.ndarray`, optional
        Third elastic modulus, 
        (use ``None`` if the data is already available)
    m1init : :obj:`numpy.ndarray`, optional
        Initial P-velocity model in m/s as starting guess for inversion
    m2init : :obj:`numpy.ndarray`, optional
        Initial S-velocity model in m/s as starting guess for inversion
    m3init : :obj:`numpy.ndarray`, optional
        Initial density model in m/s as starting guess for inversion
    m1range : :obj:`tuple`, optional
        Velocity range (min, max) to be used in loss and gradient computations
        (can be provided instead of ``m1`` to create a propagator for ``m1init``
        with a time axis that is however consistent with that of the data modelled with ``m1``)
    m2range : :obj:`tuple`, optional
        Velocity range (min, max) to be used in loss and gradient computations
        (can be provided instead of ``m2`` to create a propagator for ``m2init``
        with a time axis that is however consistent with that of the data modelled with ``m2``)
    m3range : :obj:`tuple`, optional
        Velocity range (min, max) to be used in loss and gradient computations
        (can be provided instead of ``m3`` to create a propagator for ``m3init``
        with a time axis that is however consistent with that of the data modelled with ``m3``)
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    src_type : :obj:`str`, optional
        Source type
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` and ``f0`` will be ignored
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
    loss_vx : :obj:`Type`, optional
        Loss object.
    loss_vz : :obj:`Type`, optional
        Loss object.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    fs : :obj:'bool', optional
        Use free surface boundary at the top of the model.
    streamer_acquisition : :obj:'bool', optional
        Update receiver locations in geometry for each source
    par_type : :obj:`str`, optional
        Elastic wave equation parameterization type
    clearcache : :obj:`bool`, optional
        Clear devito cache (``True``) or not (``False``) after every modelling step  
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
        m1: Optional[NDArray] = None,
        m2: Optional[NDArray] = None,
        m3: Optional[NDArray] = None,
        m1init: Optional[NDArray] = None,
        m2init: Optional[NDArray] = None,
        m3init: Optional[NDArray] = None,
        m1range: Optional[Tuple] = None,
        m2range: Optional[Tuple] = None,
        m3range: Optional[Tuple] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        checkpointing: Optional[bool] = False,
        loss_vx: Optional[Type] = None,
        loss_vz: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
        base_comm: Optional[MPIType] = None,
        fs: Optional[bool] = False,
        streamer_acquisition: Optional[bool] = False,
        par_type: Optional[str] = "vp-vs-rho",
        data: Optional[list] = None,
        clearcache: Optional[bool] = False,
    ) -> None:

        # Create m1 if not provided and m1range is available
        if m1 is None and m1range is not None:
            m1 = m1range[0] * np.ones(shape)
            m1[-1, -1] = m1range[1]
            m2 = m2range[0] * np.ones(shape)
            m2[-1, -1] = m2range[1]
            m3 = m3range[0] * np.ones(shape)
            m3[-1, -1] = m3range[1]

        # Velocity checks to ensure either m1 or vint are provided
        if (m1 is None and m1init is None) or (m2 is None and m2init is None) or (m3 is None and m3init is None):
            raise ValueError("Either m1 or m1init must be provided...")

        # Modelling parameters
        self.space_order = space_order
        self.nbl = nbl
        self.fs = fs
        self.streamer_acquisition = streamer_acquisition
        self.checkpointing = checkpointing
        self.wav = wav
        self.par_type = par_type
        self.clearcache = clearcache

        # Inversion parameters
        self.loss_vx = loss_vx
        self.loss_vz = loss_vz
        self.losshistory = []

        self.nsrc = len(src_x)

        # MPI parameters
        self.base_comm = base_comm
        
        # Create model
        self.modelexists = True if m1 is not None else False

        if m1init is not None:
            self.initmodel = self._create_model(shape, origin, spacing, m1init, m2init, m3init, space_order, nbl, fs)
        if m1 is not None:
            self.model = self._create_model(shape, origin, spacing, m1, m2, m3, space_order, nbl, fs)

        # Create geometry
        self.geometry = self._create_geometry(self.model if m1 is not None else self.initmodel,
                                              src_x, src_z, rec_x, rec_z, t0, tn, src_type,
                                              f0=f0, dt=dt)
        self.geometry1shot = self._create_geometry(self.model if m1 is not None else self.initmodel,
                                                   src_x[:1], src_z[:1], rec_x, rec_z, t0, tn, src_type,
                                                   f0=f0, dt=dt)

        if data is not None:
            self.data_vx_obs  = data[0]
            self.data_vz_obs  = data[1]

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
        m1: NDArray,
        m2: NDArray,
        m3: NDArray,
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
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        m1 : :obj:`numpy.ndarray`
            First elastic modulus
        m2 : :obj:`numpy.ndarray`
            Second elastic modulus
        m3 : :obj:`numpy.ndarray`
            Third elastic modulus
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
        model = ElasticSeismicModel(
            space_order=space_order,
            vp=m1,
            vs=m2,
            rho=m3,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl, fs=fs
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
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time in ms
        tn : :obj:`float`
            Final time in ms
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)
        dt : :obj:`float`, optional
            Time step time in ms (if provided, the geometry time_axis is
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
            f0=None if f0 is None else f0 * 1e-3,
            fs=self.model.fs if self.modelexists else self.initmodel.fs,
        )

        # Resample geometry to user defined dt
        if dt is not None:
            geometry.resample(dt)

        return geometry

    def _mod_oneshot(self, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        d_vx : :obj:`np.ndarray`
            Data of size ``nr \times nt``
        d_vz : :obj:`np.ndarray`
            Data of size ``nr \times nt``

        """
        # Update source location in geometry
        geometry = self.geometry1shot
        geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]
        if self.streamer_acquisition:
            # Update receiver locations in geometry
            geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + geometry.rec_positions[:, 0]

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid if self.modelexists else self.initmodel.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # Create data object
        rec_vx = Receiver(name='rec_vx', grid=self.model.grid if self.modelexists else self.initmodel.grid, 
                          time_range=geometry.time_axis,
                          npoint=geometry.nrec, coordinates=geometry.rec_positions)
        rec_vz = Receiver(name='rec_vz', grid=self.model.grid if self.modelexists else self.initmodel.grid, 
                          time_range=geometry.time_axis,
                          npoint=geometry.nrec, coordinates=geometry.rec_positions)
        
        # Solve
        solver = ElasticWaveSolver(self.model if self.modelexists else self.initmodel, geometry, 
                                    space_order=self.space_order)

        _, d_vx, d_vz, _, _, _ = solver.forward(
            src=src, rec_vx=rec_vx, rec_vz=rec_vz,
            model=self.model if self.modelexists else self.initmodel, par=self.par_type
        )

        # Resample
        if dt is None:
            d_vx = d_vx.data.copy()
            d_vz = d_vz.data.copy()
        else:
            d_vx = d_vx.resample(dt).data.copy()
            d_vz = d_vz.resample(dt).data.copy()
        return d_vx, d_vz

    def mod_allshots(self, dt=None) -> NDArray:
        """FD modelling for all shots

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        nsrc = self.geometry.src_positions.shape[0]
        d_vx_tot = []
        d_vz_tot = []

        for isrc in range(nsrc):
            d_vx, d_vz = self._mod_oneshot(isrc, dt)
            d_vx_tot.append(d_vx)
            d_vz_tot.append(d_vz)
            
            if self.clearcache:
                clear_devito_cache()
        d_vx_tot = np.array(d_vx_tot).reshape(nsrc, d_vx.shape[0], d_vx.shape[1])
        d_vz_tot = np.array(d_vz_tot).reshape(nsrc, d_vz.shape[0], d_vz.shape[1])
        
        return d_vx_tot, d_vz_tot

    def mod_allshots_mpi(self, dt=None) -> NDArray:
        """FD modelling for all shots with mpi gathering

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data for all shots

        """
        d_vx_totrank, d_vz_totrank = self.mod_allshots(dt)

        # gather shots from all ranks
        d_vx_tot = np.concatenate(self.base_comm.allgather(d_vx_totrank), axis=0)
        d_vz_tot = np.concatenate(self.base_comm.allgather(d_vz_totrank), axis=0)
        
        return d_vx_tot, d_vz_tot

    def _adjoint_source(self, d_vx_syn, d_vz_syn, isrc):
        """Adjoint source computation

        Note to self, takes flatten inputs and returns flatten outputs
        """
        if self.loss_vz is None:
            return self.loss_vx.grad(d_vx_syn, d_vz_syn, isrc)
        else:
            return self.loss_vx.grad(d_vx_syn, isrc), self.loss_vz.grad(d_vz_syn, isrc)

    def _loss_grad_oneshot(self, initmodel, src, solver, d_vx_syn, d_vz_syn, 
                           adjsrc_vx, adjsrc_vz, grad_1, grad_2, grad_3, isrc,
                           computeloss=True, computegrad=True, debug=False) -> Tuple[float, NDArray]:
        """Raw loss function and gradient for one shot

        Compute raw loss function and gradient for one shot without applying any pre/post-processing. Note
        that Devito returns the gradient for slowness square.

        """
        # Compute synthetic data and full forward wavefield u0
        _, _, _, v0, _, _ = solver.forward(
            src=src, rec_vx=d_vx_syn, rec_vz=d_vz_syn,
            model=initmodel, par=self.par_type, save=True
        )

        # Compute loss
        if computeloss:
            if self.loss_vz is None:
                loss = self.loss_vx(d_vx_syn.data[:].ravel(), d_vz_syn.data[:].ravel(), isrc)
            else:
                loss = 0.5*self.loss_vx(d_vx_syn.data[:].ravel(), isrc) + 3*0.5*self.loss_vz(d_vz_syn.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            _adjsrc_vx, _adjsrc_vz = self._adjoint_source(d_vx_syn.data[:].ravel(), d_vz_syn.data[:].ravel(), isrc)
            adjsrc_vx.data[:] = _adjsrc_vx.reshape(adjsrc_vx.data.shape)
            adjsrc_vz.data[:] = _adjsrc_vz.reshape(adjsrc_vz.data.shape)

            # Compute gradient
            solver.gradient(rec_vx=adjsrc_vx, rec_vz=adjsrc_vz, v=v0, par=self.par_type, 
                            grad1=grad_1, grad2=grad_2, grad3=grad_3,
                            model=initmodel, checkpointing=self.checkpointing)
        
        if computeloss and computegrad:
            return loss, grad_1, grad_2, grad_3
        elif computeloss:
            return loss
        else:
            return grad_1, grad_2, grad_3
        
    def _loss_grad(self, initmodel, isrcs=None, postprocess1=None, postprocess2=None, postprocess3=None, 
                   computeloss=True, computegrad=True, debug=False):
        """Compute loss function and gradient
        
        Parameters
        ----------
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
            nsrc = self.geometry.src_positions.shape[0]
            isrcs = range(nsrc)

        # Geometry for single source
        geometry = self.geometry1shot

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid if self.modelexists else self.initmodel.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)

        # Solver
        solver = ElasticWaveSolver(self.model if self.modelexists else self.initmodel,
                                    geometry,
                                    space_order=self.space_order)
        
        # Symbols to hold the observed data, modelled data, adjoint source, and gradient
        d_vx_syn = Receiver(name='d_vx_syn', grid=self.initmodel.grid,
                         time_range=geometry.time_axis, 
                         coordinates=geometry.rec_positions)
        d_vz_syn = Receiver(name='d_vz_syn', grid=self.initmodel.grid,
                         time_range=geometry.time_axis, 
                         coordinates=geometry.rec_positions)
        adjsrc_vx = Receiver(name='adjsrc_vx', grid=self.initmodel.grid,
                          time_range=geometry.time_axis, 
                          coordinates=geometry.rec_positions)
        adjsrc_vz = Receiver(name='adjsrc_vz', grid=self.initmodel.grid,
                          time_range=geometry.time_axis, 
                          coordinates=geometry.rec_positions)
        grad_1 = Function(name="grad_1", grid=self.initmodel.grid)
        grad_2 = Function(name="grad_2", grid=self.initmodel.grid)
        grad_3 = Function(name="grad_3", grid=self.initmodel.grid)

        # Compute loss and gradient
        loss = 0.
        for isrc in isrcs:
            # Update source location in geometry
            geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]
            if self.streamer_acquisition:
                # Update receiver locations in geometry
                geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + geometry.rec_positions[:, 0]         
                d_vx_syn = Receiver(name='d_vx_syn', grid=self.initmodel.grid,
                                time_range=geometry.time_axis, 
                                coordinates=geometry.rec_positions)
                d_vz_syn = Receiver(name='d_vz_syn', grid=self.initmodel.grid,
                                time_range=geometry.time_axis, 
                                coordinates=geometry.rec_positions)
                adjsrc_vx = Receiver(name='adjsrc_vx', grid=self.initmodel.grid,
                                time_range=geometry.time_axis, 
                                coordinates=geometry.rec_positions)
                adjsrc_vz = Receiver(name='adjsrc_vz', grid=self.initmodel.grid,
                                time_range=geometry.time_axis, 
                                coordinates=geometry.rec_positions)
                
            # Compute loss and gradient for one shot
            if debug:
                if isrc == self.nsrc//2:
                    debug_isrc = True
                else:
                    debug_isrc = False
            else:
                debug_isrc = False
                
            lossgrad = self._loss_grad_oneshot(initmodel, src, solver, d_vx_syn, d_vz_syn, 
                                               adjsrc_vx, adjsrc_vz, grad_1, grad_2, grad_3, isrc,
                                               computeloss=computeloss, computegrad=computegrad, debug=debug_isrc)
            
            if computeloss and computegrad:
                loss += lossgrad[0]
            elif computeloss:
                loss += lossgrad

        if computegrad:
            grad_1 = grad_1.data[:]
            grad_2 = grad_2.data[:]
            grad_3 = grad_3.data[:]
            
        if self.clearcache:
            clear_devito_cache()

        # Gather gradients 
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad_1 = self.base_comm.allreduce(grad_1, op=MPI.SUM)
                grad_2 = self.base_comm.allreduce(grad_2, op=MPI.SUM)
                grad_3 = self.base_comm.allreduce(grad_3, op=MPI.SUM)

        # Postprocess loss and gradient
        grad_1 = self._crop_model(grad_1, self.nbl, self.fs)
        grad_2 = self._crop_model(grad_2, self.nbl, self.fs)
        grad_3 = self._crop_model(grad_3, self.nbl, self.fs)

        if self.par_type=='vp-vs-rho':
            m1 = self._crop_model(initmodel.vp.data[:], self.nbl, self.fs)
            m2 = self._crop_model(initmodel.vs.data[:], self.nbl, self.fs)
            m3 = self._crop_model(initmodel.rho.data[:], self.nbl, self.fs)
        elif self.par_type=='Ip-Is-rho':
            m1 = self._crop_model(initmodel.Ip.data[:], self.nbl, self.fs)
            m2 = self._crop_model(initmodel.Is.data[:], self.nbl, self.fs)
            m3 = self._crop_model(initmodel.rho.data[:], self.nbl, self.fs)
        elif self.par_type=='lam-mu':
            m1 = self._crop_model(initmodel.lam.data[:], self.nbl, self.fs)
            m2 = self._crop_model(initmodel.mu.data[:], self.nbl, self.fs)
            m3 = self._crop_model(initmodel.rho.data[:], self.nbl, self.fs)
        
        if postprocess1 is not None:
            _, grad_1 = postprocess1(m1, loss, grad_1)
            _, grad_2 = postprocess2(m2, loss, grad_2)
            _, grad_3 = postprocess3(m3, loss, grad_3)
            
        if computeloss and computegrad:
            return loss, grad_1, grad_2, grad_3
        elif computeloss:
            return loss
        else:
            return grad_1, grad_2, grad_3
        
    def loss_grad(self, x, y, z, convertX=None, 
                  postprocess1=None, postprocess2=None, postprocess3=None,
                  computeloss=True, computegrad=True, debug=False):
        """Compute loss function and gradient to be used by solver

        This routine wraps the _loss_grad providing and returning numpy arrays 
        and should be used with any solver
        
        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertX : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
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

        # Convert x to velocity
        if convertX is None:
            m1 = x.reshape(self.initmodel.shape)
            m2 = y.reshape(self.initmodel.shape)
            m3 = z.reshape(self.initmodel.shape)
        else:
            m1 = convertX(x.reshape(self.initmodel.shape))
            m2 = convertX(y.reshape(self.initmodel.shape))
            m3 = convertX(z.reshape(self.initmodel.shape))

        # Overwrite current velocity in devito model used to compute the synthetic data
        if self.par_type=='vp-vs-rho':
            self.initmodel.update('vp', m1.reshape(self.initmodel.shape))
            self.initmodel.update('vs', m2.reshape(self.initmodel.shape))
            self.initmodel.update('rho', m3.reshape(self.initmodel.shape))
        elif self.par_type=='Ip-Is-rho':
            self.initmodel.update('Ip', m1.reshape(self.initmodel.shape))
            self.initmodel.update('Is', m2.reshape(self.initmodel.shape))
            self.initmodel.update('rho', m3.reshape(self.initmodel.shape))
        elif self.par_type=='lam-mu':
            self.initmodel.update('lam', m1.reshape(self.initmodel.shape))
            self.initmodel.update('mu', m2.reshape(self.initmodel.shape))
            self.initmodel.update('rho', m3.reshape(self.initmodel.shape))
            
        # Evaluate objective function and gradient
        lossgrad = self._loss_grad(self.initmodel,
                                   postprocess1=postprocess1, postprocess2=postprocess2, 
                                   postprocess3=postprocess3,
                                   computeloss=computeloss,
                                   computegrad=computegrad, debug=debug)

        # Split lossgrad based on what has been computed in self._loss_grad
        if computeloss and computegrad:
            loss, grad_1, grad_2, grad_3 = lossgrad
        elif computeloss:
            loss, grad_1, grad_2, grad_3 = lossgrad, None, None, None
        else:
            loss, grad_1, grad_2, grad_3 = None, lossgrad

        # Save loss history
        if computeloss:
            self.losshistory.append(loss)

        # Return loss, grad or both
        if computeloss and computegrad:
            return loss, grad_1.ravel(), grad_2.ravel(), grad_3.ravel()
        elif computeloss:
            return loss
        else:
            return grad_1.ravel(), grad_2.ravel(), grad_3.ravel()

    def loss(self, x, y, z, convertX=None, postprocess1=None, postprocess2=None, postprocess3=None):
        """Compute loss function to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertX : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        loss : :obj:`float`
            Loss function

        """
        return self.loss_grad(x, y, z, convertX=convertX, 
                              postprocess1=postprocess1, postprocess2=postprocess2, postprocess3=postprocess3,
                              computeloss=True, computegrad=False)

    def grad(self, x, y, z, convertX=None, postprocess1=None, postprocess2=None, postprocess3=None):
        """Compute gradient to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertX : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """
        return self.loss_grad(x, y, z, convertX=convertX, 
                              postprocess1=postprocess1, postprocess2=postprocess2, postprocess3=postprocess3,
                              computeloss=False, computegrad=True)