from devito.tools import memoized_meth
from devito import VectorTimeFunction, TensorTimeFunction, Function
from examples.seismic import PointSource
from devito import (Eq, Operator, VectorTimeFunction, TensorTimeFunction,
                    Function, TimeFunction)
from devito import solve
from examples.seismic import PointSource, Receiver
from devito.types.tensor import (TensorFunction, TensorTimeFunction,
                                 VectorFunction, VectorTimeFunction, tens_func)
import numpy as np

from sympy import symbols, Matrix, ones
from examples.seismic import SeismicModel

class ElasticSeismicModel(SeismicModel):

    def _initialize_physics(self, vp, space_order, **kwargs):

        params = []
        # Buoyancy
        rho = kwargs.get('rho', 1)
        self.rho = self._gen_phys_param(rho, 'rho', space_order)

        # Initialize elastic with Lame parametrization
        try:
            vs = kwargs.pop('vs')
        except:
            raise Exception("ElasticSeismicModel must receive 'vs' as an argument")

        self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)*rho, 'lam', space_order,
                                        is_param=True)
        self.mu = self._gen_phys_param((vs**2) * rho, 'mu', space_order, is_param=True)
        self.vs = self._gen_phys_param(vs, 'vs', space_order)
        self.vp = self._gen_phys_param(vp, 'vp', space_order)

        self.Ip = self._gen_phys_param(vp*rho, 'Ip', space_order, is_param=True)
        self.Is = self._gen_phys_param(vs*rho, 'Is', space_order, is_param=True)

        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

class C_Matrix():

    C_matrix_dependency = {'lam-mu': 'C_lambda_mu', 'vp-vs-rho': 'C_vp_vs_rho',
                           'Ip-Is-rho': 'C_Ip_Is_rho'}

    def __new__(cls, model, parameters):
        c_m_gen = cls.C_matrix_gen(parameters)
        return c_m_gen(model)

    @classmethod
    def C_matrix_gen(cls, parameters):
        return getattr(cls, cls.C_matrix_dependency[parameters])

    def _matrix_init(dim):
        def cij(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii == jj or (ii <= dim and jj <= dim)):
                return symbols('C%s%s' % (ii, jj))
            return 0

        d = dim*2 + dim-2
        Cij = [[cij(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Cij)

    @classmethod
    def C_lambda_mu(cls, model):
        def subs3D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': lmbda + (2*mu),
                    'C44': mu,
                    'C55': mu,
                    'C66': mu,
                    'C12': lmbda,
                    'C13': lmbda,
                    'C23': lmbda}

        def subs2D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': mu,
                    'C12': lmbda}

        matriz = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matriz.subs(subs)

        M.dlam = cls._generate_Dlam(model)
        M.dmu = cls._generate_Dmu(model)
        M.inv = cls._inverse_C_lam(model)
        return M

    @staticmethod
    def _inverse_C_lam(model):
        def subs3D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C44': 1/mu,
                    'C55': 1/mu,
                    'C66': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C13': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C23': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        def subs2D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        matrix = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @staticmethod
    def _generate_Dlam(model):
        def d_lam(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return 1
            return 0

        d = model.dim*2 + model.dim-2
        Dlam = [[d_lam(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dlam)

    @staticmethod
    def _generate_Dmu(model):
        def d_mu(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii == jj):
                if ii <= model.dim:
                    return 2
                else:
                    return 1
            return 0

        d = model.dim*2 + model.dim-2
        Dmu = [[d_mu(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dmu)

    @classmethod
    def C_vp_vs_rho(cls, model):
        def subs3D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vp*vp,
                    'C44': rho*vs*vs,
                    'C55': rho*vs*vs,
                    'C66': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs,
                    'C13': rho*vp*vp - 2*rho*vs*vs,
                    'C23': rho*vp*vp - 2*rho*vs*vs}

        def subs2D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = model.rho

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        M.dvp = cls._generate_Dvp(model)
        M.dvs = cls._generate_Dvs(model)
        M.drho = cls._generate_Drho(model)
        M.inv = cls._inverse_C_vp_vs(model)
        return M

    @staticmethod
    def _inverse_C_vp_vs(model):
        def subs3D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C44': 1/(rho*vs*vs),
                    'C55': 1/(rho*vs*vs),
                    'C66': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C13': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C23': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        def subs2D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = model.rho

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @staticmethod
    def _generate_Dvp(model):
        def d_vp(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return 2*model.rho*model.vp
            return 0

        d = model.dim*2 + model.dim-2
        Dvp = [[d_vp(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dvp)

    @staticmethod
    def _generate_Dvs(model):
        def subs3D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 0,
                    'C44': 2*rho*vs,
                    'C55': 2*rho*vs,
                    'C66': 2*rho*vs,
                    'C12': -4*rho*vs,
                    'C13': -4*rho*vs,
                    'C23': -4*rho*vs}

        def subs2D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 2*rho*vs,
                    'C12': -4*rho*vs}

        Dvs = C_Matrix._matrix_init(model.dim)
        rho = model.rho
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return Dvs.subs(subs)

    @staticmethod
    def _generate_Drho(model):
        def subs3D():
            return {'C11': vp*vp,
                    'C22': vp*vp,
                    'C33': vp*vp,
                    'C44': vs*vs,
                    'C55': vs*vs,
                    'C66': vs*vs,
                    'C12': vp*vp - 2*vs*vs,
                    'C13': vp*vp - 2*vs*vs,
                    'C23': vp*vp - 2*vs*vs}

        def subs2D():
            return {'C11': vp*vp,
                    'C22': vp*vp,
                    'C33': vs*vs,
                    'C12': vp*vp - 2*vs*vs}

        Dvs = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return Dvs.subs(subs)

    @classmethod
    def C_Ip_Is_rho(cls, model):
        def subs3D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Ip*vp,
                    'C44': Is*vs,
                    'C55': Is*vs,
                    'C66': Is*vs,
                    'C12': Ip*vp - 2*Is*vs,
                    'C13': Ip*vp - 2*Is*vs,
                    'C23': Ip*vp - 2*Is*vs}

        def subs2D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Is*vs,
                    'C12': Ip*vp - 2*Is*vs}

        matrix = cls._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        Ip = model.Ip
        Is = model.Is

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        M.dIs = cls._generate_DIs(model)
        M.dIp = cls._generate_DIp(model)

        return M

    @staticmethod
    def _generate_DIp(model):
        def d_Ip(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return model.vp
            return 0

        d = model.dim*2 + model.dim-2
        D_Ip = [[d_Ip(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(D_Ip)

    @staticmethod
    def _generate_DIs(model):
        def subs3D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 0,
                    'C44': vs,
                    'C55': vs,
                    'C66': vs,
                    'C12': -2*vs,
                    'C13': -2*vs,
                    'C23': -2*vs}

        def subs2D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': vs,
                    'C12': -2*vs}

        D_Is = C_Matrix._matrix_init(model.dim)
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return D_Is.subs(subs)


def D(self, shift=None):
    """
    Returns the result of matrix D applied over the TensorFunction.
    """
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")

    M = tensor(self) if self.shape[0] != self.shape[1] else self

    comps = []
    func = tens_func(self)
    for j, d in enumerate(self.space_dimensions):
        comps.append(sum([getattr(M[j, i], 'd%s' % d.name)
                         for i, d in enumerate(self.space_dimensions)]))
    return func._new(comps)


def S(self, shift=None):
    """
    Returns the result of transposed matrix D applied over the VectorFunction.
    """
    if not self.is_VectorValued:
        raise TypeError("The object must be a Vector object")

    derivs = ['d%s' % d.name for d in self.space_dimensions]

    comp = []
    comp.append(getattr(self[0], derivs[0]))
    comp.append(getattr(self[1], derivs[1]))
    if len(self.space_dimensions) == 3:
        comp.append(getattr(self[2], derivs[2]))
        comp.append(getattr(self[1], derivs[2]) + getattr(self[2], derivs[1]))
        comp.append(getattr(self[0], derivs[2]) + getattr(self[2], derivs[0]))
    comp.append(getattr(self[0], derivs[1]) + getattr(self[1], derivs[0]))

    func = tens_func(self)

    return func._new(comp)


def vec(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] != self.shape[1]:
        raise Exception("This object is already represented by its vector form.")

    order = ([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
             if len(self.space_dimensions) == 3 else [(0, 0), (1, 1), (0, 1)])
    comp = [self[o[0], o[1]] for o in order]
    func = tens_func(self)
    return func(comp)

def tensor(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] == self.shape[1]:
        raise Exception("This object is already represented by its tensor form.")

    ndim = len(self.space_dimensions)
    M = np.zeros((ndim, ndim), dtype=np.dtype(object))
    M[0, 0] = self[0]
    M[1, 1] = self[1]
    if len(self.space_dimensions) == 3:
        M[2, 2] = self[2]
        M[2, 1] = self[3]
        M[1, 2] = self[3]
        M[2, 0] = self[4]
        M[0, 2] = self[4]
    M[1, 0] = self[-1]
    M[0, 1] = self[-1]

    func = tens_func(self)
    return func._new(M)

def gather(a1, a2):

    expected_a1_types = [int, VectorFunction, VectorTimeFunction]
    expected_a2_types = [int, TensorFunction, TensorTimeFunction]

    if type(a1) not in expected_a1_types:
        raise ValueError("a1 must be a VectorFunction or a Integer")
    if type(a2) not in expected_a2_types:
        raise ValueError("a2 must be a TensorFunction or a Integer")
    if type(a1) is int and type(a2) is int:
        raise ValueError("Both a2 and a1 cannot be Integers simultaneously")

    if type(a1) is int:
        a1_m = Matrix([ones(len(a2.space_dimensions), 1)*a1])
    else:
        a1_m = Matrix(a1)

    if type(a2) is int:
        ndim = len(a1.space_dimensions)
        a2_m = Matrix([ones((3*ndim-3), 1)*a2])
    else:
        a2_m = Matrix(a2)

    if a1_m.cols > 1:
        a1_m = a1_m.T
    if a2_m.cols > 1:
        a2_m = a2_m.T

    return Matrix.vstack(a1_m, a2_m)

def src_rec(v, tau, model, geometry, forward=True):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    # src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
    #                   npoint=geometry.nsrc)
    src = geometry.src # added by @hatsyim
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)
    name = "rec_tau" if forward else "rec"
    rec = Receiver(name="%s" % name, grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)
    tau = vec(tau)
    if forward:

        # The source injection term
        src_xx = src.inject(field=tau[0].forward, expr=src * s)
        src_zz = src.inject(field=tau[1].forward, expr=src * s)
        src_expr = src_xx + src_zz
        if model.grid.dim == 3:
            src_yy = src.inject(field=tau[2].forward, expr=src * s)
            src_expr += src_yy
        # Create interpolation expression for receivers
        rec_term_vx = rec_vx.interpolate(expr=v[0])
        rec_term_vz = rec_vz.interpolate(expr=v[-1])
        expr = tau[0] + tau[1]
        rec_expr = rec_term_vx + rec_term_vz
        if model.grid.dim == 3:
            expr += tau[2]
            rec_term_vy = rec_vy.interpolate(expr=v[1])
            rec_expr += rec_term_vy
        rec_term_tau = rec.interpolate(expr=expr)
        rec_expr += rec_term_tau

    else:
        # Construct expression to inject receiver values
        rec_xx = rec.inject(field=tau[0].backward, expr=rec*s)
        rec_zz = rec.inject(field=tau[1].backward, expr=rec*s)
        rec_expr = rec_xx + rec_zz
        expr = tau[0] + tau[1]
        if model.grid.dim == 3:
            rec_expr += rec.inject(field=tau[2].backward, expr=rec*s)
            expr += tau[2]
        # Create interpolation expression for the adjoint-source
        src_expr = src.interpolate(expr=expr)

    return src_expr, rec_expr

def elastic_stencil(model, v, tau, forward=True, par='lam-mu'):

    damp = model.damp

    rho = model.rho

    C = C_Matrix(model, par)

    tau = vec(tau)
    if forward:

        pde_v = rho * v.dt - D(tau)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        pde_tau = tau.dt - C * S(v.forward)
        u_t = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

        return [u_v, u_t]

    else:

        """
        Implementation of the elastic wave-equation from:
        1 - Feng and Schuster (2017): Elastic least-squares reverse time migration
        https://doi.org/10.1190/geo2016-0254.1
        """

        pde_v = rho * v.dtl - D(C.T*tau)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        pde_tau = -tau.dtl + S(v.backward)
        u_t = Eq(tau.backward, damp * solve(pde_tau, tau.backward))

        return [u_v, u_t]


def EqsLamMu(model, sig, u, v, grad_lam, grad_mu, grad_rho, C, space_order=8):
    hl = TimeFunction(name='hl', grid=model.grid, space_order=space_order,
                      time_order=1)
    hm = TimeFunction(name='hm', grid=model.grid, space_order=space_order,
                      time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wl = gather(0, C.dlam * S(v))
    Wm = gather(0, C.dmu * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wl_update = Eq(hl, Wl.T * W2)
    gradient_lam = Eq(grad_lam, grad_lam + hl)

    wm_update = Eq(hm, Wm.T * W2)
    gradient_mu = Eq(grad_mu, grad_mu + hm)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wl_update, gradient_lam, wm_update, gradient_mu, wr_update, gradient_rho]


def EqsVpVsRho(model, sig, u, v, grad_vp, grad_vs, grad_rho, C, space_order=8):
    hvp = TimeFunction(name='hvp', grid=model.grid, space_order=space_order,
                       time_order=1)
    hvs = TimeFunction(name='hvs', grid=model.grid, space_order=space_order,
                       time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wvp = gather(0, -C.dvp * S(v))
    Wvs = gather(0, -C.dvs * S(v))
    Wr = gather(v.dt, - C.drho * S(v))

    W2 = gather(u, sig)

    wvp_update = Eq(hvp, Wvp.T * W2)
    gradient_lam = Eq(grad_vp, grad_vp - hvp)

    wvs_update = Eq(hvs, Wvs.T * W2)
    gradient_mu = Eq(grad_vs, grad_vs - hvs)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wvp_update, gradient_lam, wvs_update, gradient_mu, wr_update, gradient_rho]


def EqsIpIs(model, sig, u, v, grad_Ip, grad_Is, grad_rho, C, space_order=8):

    hIp = TimeFunction(name='hIp', grid=model.grid, space_order=space_order,
                       time_order=1)

    hIs = TimeFunction(name='hIs', grid=model.grid, space_order=space_order,
                       time_order=1)

    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    WIp = gather(0, C.dIp * S(v))
    WIs = gather(0, C.dIs * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wIp_update = Eq(hIp, WIp.T * W2)
    gradient_Ip = Eq(grad_Ip, grad_Ip + hIp)

    wIs_update = Eq(hIs, WIs.T * W2)
    gradient_Is = Eq(grad_Is, grad_Is + hIs)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wIp_update, gradient_Ip, wIs_update, gradient_Is, wr_update, gradient_rho]

def ForwardOperator(model, geometry, space_order=4, save=False, par='lam-mu', **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             space_order=space_order, time_order=1)

    eqn = elastic_stencil(model, v, tau, par=par)

    src_expr, rec_expr = src_rec(v, tau, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name="ForwardElastic", **kwargs)


def AdjointOperator(model, geometry, space_order=4, par='lam-mu', **kwargs):
    """
    Construct an adjoint modelling operator in a viscoacoustic medium.
    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """

    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)

    src_expr, rec_expr = src_rec(u, sig, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name='AdjointElastic', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True, par='lam-mu', **kwargs):
    """
    Construct a gradient operator in an elastic media.
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
        Option to store the entire (unrolled) wavefield.
    """
    # Gradient symbol and wavefield symbols
    grad1 = Function(name='grad1', grid=model.grid)
    grad2 = Function(name='grad2', grid=model.grid)
    grad3 = Function(name='grad3', grid=model.grid)

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)
    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)

    s = model.grid.time_dim.spacing
    rho = model.rho

    C = C_Matrix(model, par)

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)
    sig = vec(sig)

    kernel = kernels[par]
    gradient_update = kernel(model, sig, u, v, grad1, grad2,
                             grad3, C, space_order=space_order)

    # Construct expression to inject receiver values
    rec_term_vx = rec_vx.inject(field=u[0].backward, expr=s*rec_vx/rho)
    rec_term_vz = rec_vz.inject(field=u[-1].backward, expr=s*rec_vz/rho)
    rec_expr = rec_term_vx + rec_term_vz
    if model.grid.dim == 3:
        rec_expr += rec_vy.inject(field=u[1].backward, expr=s*rec_vy/rho)

    if kwargs.pop('has_rec_p'):
        rec_p = Receiver(name='rec_p', grid=model.grid, time_range=geometry.time_axis,
                         npoint=geometry.nrec)
        rec_term_sigx = rec_p.inject(field=sig[0].backward, expr=s*rec_p/rho)
        rec_term_sigz = rec_p.inject(field=sig[1].backward, expr=s*rec_p/rho)
        rec_expr += rec_term_sigx + rec_term_sigz
        if model.grid.dim == 3:
            rec_expr += rec_p.inject(field=sig[2].backward, expr=rec_p/rho)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + rec_expr + gradient_update, subs=model.spacing_map,
                    name='GradientElastic', **kwargs)


kernels = {'lam-mu': EqsLamMu, 'vp-vs-rho': EqsVpVsRho, 'Ip-Is-rho': EqsIpIs}

class ElasticWaveSolver(object):
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
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry

        self.space_order = space_order
        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None, par=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, par=par, **self._kwargs)

    @memoized_meth
    def op_adj(self, par=None):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order, par=par, **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True, par=None, has_rec_p=None):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                space_order=self.space_order, par=par,
                                has_rec_p=has_rec_p, **self._kwargs)

    def forward(self, src=None, rec_tau=None, rec_vx=None, rec_vz=None, rec_vy=None,
                v=None, tau=None, model=None, save=None, par='lam-mu', **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.
        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec_tau : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the sum of the tensor component.
        rec_vx : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the x component of particle velocities.
        rec_vy : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the y compenent of particle velocities.
        rec_vz : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the z compenent of particle velocities.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        tau : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        model : Model, optional
            Object containing the physical parameters.
        lam : Function, optional
            The time-constant first Lame parameter `rho * (vp**2 - 2 * vs **2)`.
        mu : Function, optional
            The Shear modulus `(rho * vs*2)`.
        b : Function, optional
            The time-constant inverse density (b=1 for water).
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.
        Returns
        -------
        rec_tau, rec_vx, rec_vy, rec_vz, particle velocities v, stress tensor tau and
        performance summary.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec_vx = rec_vx or self.geometry.new_rec(name='rec_vx')
        rec_vz = rec_vz or self.geometry.new_rec(name='rec_vz')
        if self.model.grid.dim == 3:
            rec_vy = rec_vy or self.geometry.new_rec(name='rec_vy')
            kwargs.update({'rec_vy': rec_vy})
        rec_tau = rec_tau or self.geometry.new_rec(name='rec_tau')

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        save_t = src.nt if save else None
        v = v or VectorTimeFunction(name='v', grid=self.model.grid, save=save_t,
                                    space_order=self.space_order, time_order=1)
        tau = tau or TensorTimeFunction(name='tau', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})

        model = model or self.model
        # Pick Lame parameters from model unless explicitly provided

        parameters = model.physical_params(**kwargs)

        # Pick specifics physical parameters from model unless explicitly provided
        new_p = {k: v for k, v in parameters.items() if k not in remove_par[par]}
        kwargs.update(new_p)

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save, par).apply(src=src, rec_tau=rec_tau,
                                               rec_vx=rec_vx, rec_vz=rec_vz,
                                               dt=kwargs.pop('dt', self.dt), **kwargs)
        if self.model.grid.dim == 3:
            return rec_tau, rec_vx, rec_vy, rec_vz, v, tau, summary
        return rec_tau, rec_vx, rec_vz, v, tau, summary

    def adjoint(self, rec, srca=None, u=None, sig=None, model=None, par='lam-mu',
                **kwargs):
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
        u : VectorTimeFunction, optional
            The computed particle velocity.
        sig : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        model : Model, optional
            Object containing the physical parameters.
        lam : Function, optional
            The time-constant first Lame parameter `rho * (vp**2 - 2 * vs **2)`.
        mu : Function, optional
            The Shear modulus `(rho * vs*2)`.
        b : Function, optional
            The time-constant inverse density (b=1 for water).
        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.geometry.time_axis,
                                   coordinates=self.geometry.src_positions)

        u = u or VectorTimeFunction(name="u", grid=self.model.grid,
                                    time_order=1, space_order=self.space_order)
        sig = sig or TensorTimeFunction(name='sig', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in u})
        kwargs.update({k.name: k for k in sig})
        kwargs['time_m'] = 0

        model = model or self.model

        parameters = model.physical_params(**kwargs)

        # Pick specifics physical parameters from model unless explicitly provided
        new_p = {k: v for k, v in parameters.items() if k not in remove_par[par]}
        kwargs.update(new_p)

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj(par=par).apply(src=srca, rec=rec,
                                             dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, u, sig, summary

    def jacobian_adjoint(self, rec_vx, rec_vz, v, u=None, sig=None, rec_vy=None,
                         grad1=None, grad2=None, grad3=None, model=None,
                         checkpointing=False, par='lam-mu', **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        p : TimeFunction
            Full wavefield `p` (created with save=True).
        pa : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        r : TimeFunction, optional
            The computed attenuation memory variable.
        va : VectorTimeFunction, optional
            The computed particle velocity.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        qp : Function, optional
            The P-wave quality factor.
        b : Function, optional
            The time-constant inverse density.

        Returns
        -------
        Gradient field and performance summary.
        """
        # Gradient symbol
        grad1 = grad1 or Function(name='grad1', grid=self.model.grid)
        grad2 = grad2 or Function(name='grad2', grid=self.model.grid)
        grad3 = grad3 or Function(name='grad3', grid=self.model.grid)

        u = u or VectorTimeFunction(name="u", grid=self.model.grid,
                                    time_order=1, space_order=self.space_order)
        sig = sig or TensorTimeFunction(name='sig', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in u})
        kwargs.update({k.name: k for k in sig})
        if self.model.grid.dim == 3:
            kwargs.update({'rec_vy': rec_vy})
        kwargs['time_m'] = 0

        model = model or self.model

        parameters = model.physical_params(**kwargs)

        # Pick specifics physical parameters from model unless explicitly provided
        new_p = {k: v for k, v in parameters.items() if k not in remove_par[par]}
        kwargs.update(new_p)

        has_rec_p = True if kwargs.get('rec_p', None) else None
        op = self.op_grad(par=par, has_rec_p=has_rec_p)
        summary = op.apply(rec_vx=rec_vx, rec_vz=rec_vz, grad1=grad1,
                           grad2=grad2, grad3=grad3,
                           dt=kwargs.pop('dt', self.dt), **kwargs)

        return grad1, grad2, grad3, summary
    
    # Backward compatibility
    gradient = jacobian_adjoint

remove_par = {'lam-mu': ['vp', 'vs', 'Ip', 'Is'], 'vp-vs-rho': ['lam', 'mu', 'Ip', 'Is'],
              'Ip-Is-rho': ['lam', 'mu']}