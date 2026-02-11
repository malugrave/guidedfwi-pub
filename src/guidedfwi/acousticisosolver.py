from devito import Function, TimeFunction, DevitoCheckpoint, CheckpointOperator, Revolver
from devito.tools import memoized_meth
from devito import Eq, Operator, Function, TimeFunction, Inc, solve, sign, ConditionalDimension
from devito.symbolics import retrieve_functions, INT, retrieve_derivatives

def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    """
    lhs, rhs = eq.args
    # Get vertical dimension and corresponding subdimension
    fsdomain = model.grid.subdomains['fsdomain']
    zfs = fsdomain.dimensions[-1]
    z = zfs.parent

    # Retrieve vertical derivatives
    dzs = {d for d in retrieve_derivatives(rhs) if z in d.dims}
    # Remove inner duplicate
    dzs = dzs - {d for D in dzs for d in retrieve_derivatives(D.expr) if z in d.dims}
    dzs = {d: d._eval_at(lhs).evaluate for d in dzs}

    # Finally get functions for evaluated derivatives
    funcs = {f for f in retrieve_functions(dzs.values())}

    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})

    # Mapper for vertical derivatives
    dzmapper = {d: v.subs(mapper) for d, v in dzs.items()}

    fs_eq = [eq.func(lhs, rhs.subs(dzmapper), subdomain=fsdomain)]
    fs_eq.append(eq.func(lhs._subs(z, 0), 0, subdomain=fsdomain))

    return fs_eq

def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1/model.m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic

def iso_stencil(field, model, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get('forward', True)
    # Define time step to be updated
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    eq_time = solve(model.m * field.dt2 - lap - q + model.damp * udt, unext)

    # Time-stepping stencil.
    eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains['physdomain'])]

    # Add free surface
    if model.fs:
        eqns.append(freesurface(model, Eq(unext, eq_time)))
    return eqns

def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', factor=None, **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    factor : int, optional
        Downsampling factor to save snapshots of the wavefield.
    """
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # Build operator equations
    equations = eqn + src_term + rec_term

    if factor:
        # Implement snapshotting
        nsnaps = (geometry.nt + factor - 1) // factor
        time_subsampled = ConditionalDimension(
            't_sub', parent=model.grid.time_dim, factor=factor)
        usnaps = TimeFunction(name='usnaps', grid=model.grid,
                              time_order=2, space_order=space_order,
                              save=nsnaps, time_dim=time_subsampled)
        # Add equation to save snapshots
        snapshot_eq = Eq(usnaps, u)
        equations += [snapshot_eq]
    else:
        usnaps = None
    # Substitute spacing terms to reduce flops
    op = Operator(equations, subs=model.spacing_map, name='Forward', **kwargs)
    if usnaps is not None:
        return op, usnaps
    else:
        return op

def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', factor=None, **kwargs):
    """
    Construct a gradient operator in an acoustic medium.
    """
    m = model.m

    # Gradient symbol
    grad = Function(name='grad', grid=model.grid)

    # Create the adjoint wavefield
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    # Add expression for receiver injection
    rec = geometry.rec
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    time = model.grid.time_dim

    if factor is not None:
        # Condition to apply gradient update only at snapshot times
        condition = Eq(time % factor, 0)
        # Create the ConditionalDimension for subsampling
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        # Define usnaps with time_subsampled as its time dimension
        nsnaps = (geometry.nt + factor - 1) // factor
        usnaps = TimeFunction(name='usnaps', grid=model.grid,
                              time_order=2, space_order=space_order,
                              save=nsnaps, time_dim=time_subsampled)
        # Gradient update without indexing usnaps
        if kernel == 'OT2':
            gradient_update = Inc(grad, - usnaps * v.dt2, implicit_dims=[time_subsampled],
                                  condition=condition)
        elif kernel == 'OT4':
            gradient_update = Inc(grad, - usnaps * v.dt2
                                  - s**2 / 12.0 * usnaps.biharmonic(m**(-2)) * v,
                                  implicit_dims=[time_subsampled],
                                  condition=condition)
    else:
        u = TimeFunction(name='u', grid=model.grid,
                         save=geometry.nt if save else None,
                         time_order=2, space_order=space_order)
        if kernel == 'OT2':
            gradient_update = Inc(grad, - u * v.dt2)
        elif kernel == 'OT4':
            gradient_update = Inc(grad, - u * v.dt2
                                  - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
            
    # Substitute spacing terms to reduce flops
    op = Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                  name='Gradient', **kwargs)
    return op

def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel)
    eqn2 = iso_stencil(U, model, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)
    
class AcousticWaveSolver:
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """

    def __init__(self, model, geometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None, factor=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               factor=factor, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True, factor=None):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                factor=factor, **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            kernel=self.kernel, space_order=self.space_order,
                            **self._kwargs)

    def forward(self, src=None, rec=None, u=None, model=None, save=None, factor=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.
        factor : int, optional
        Downsampling factor to save snapshots of the wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        # Get the operator
        op_fwd = self.op_fwd(save=save, factor=factor)
        # Prepare parameters for operator apply
        op_args = {'src': src, 'rec': rec, 'u': u, 'dt': kwargs.pop('dt', self.dt)}
        op_args.update(kwargs)

        # Execute operator and return wavefield and receiver data
        if factor:
            # Operator returned is op, usnaps
            op, usnaps = op_fwd
            op_args['usnaps'] = usnaps
            summary = op.apply(**op_args)
            
        else:
            op = op_fwd
            usnaps = None
            summary = op.apply(**op_args)
        return rec, u, usnaps, summary

    def adjoint(self, rec, srca=None, v=None, model=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def jacobian_adjoint(self, rec, u=None, usnaps=None, src=None, v=None, grad=None, model=None,
                         factor=None, checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        usnaps : TimeFunction
            Snapshots of the wavefield `u`.
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        checkpointing : boolean, optional 
            Flag to enable checkpointing (default False). 
            Cannot be used with snapshotting.
        factor : int, optional
            Downsampling factor for the saved snapshots of the wavefield `u`.
            Cannot be used with checkpointing.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        # Check that snapshotting and checkpointing are not used together
        if factor is not None and checkpointing:
            raise ValueError("Cannot use snapshotting (factor) and checkpointing simultaneously.")

        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False),
                                         src=src or self.geometry.src,
                                         u=u, dt=dt, **kwargs)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          rec=rec, dt=dt, grad=grad, **kwargs)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            if factor is not None:
                # Get the gradient operator
                op = self.op_grad(save=False, factor=factor)
                op_args = {'rec': rec, 'grad': grad, 'v': v, 'dt': dt, 'usnaps': usnaps}
            else:
                op = self.op_grad(save=True, factor=None)
                op_args = {'rec': rec, 'grad': grad, 'v': v, 'dt': dt, 'u': u}

            op_args.update(kwargs)
            summary = op.apply(**op_args)

        return grad, summary

    def jacobian(self, dmin, src=None, rec=None, u=None, U=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    # Backward compatibility
    born = jacobian
    gradient = jacobian_adjoint