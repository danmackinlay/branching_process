have_autograd = False

try:
    import autograd
    have_autograd = True
    import autograd.numpy as np
    import autograd.scipy as sp
except ImportError as e:
    import numpy as np
    import scipy as sp


class InfluenceKernel(object):
    def __init__(
            self,
            n_kernels=1,
            *args, **fixed_args):
        self._fixed_args = fixed_args
        self._fixed_args.setdefault('kappa', np.ones(n_kernels)/n_kernels)
        super(InfluenceKernel, self).__init__(*args)

    def majorant(
            self,
            t,
            *args, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        return np.sum(
            self.majorant_each(t, *args, **new_kwargs),
            0
        )

    def __call__(
            self,
            t,
            *args, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        return np.sum(
            self.call_each(t, *args, **new_kwargs),
            0
        )

    def integrate(
            self,
            t,
            *args, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        return np.sum(
            self.integrate_each(t, *args, **new_kwargs),
            0
        )

    def majorant_each(
            self,
            t,
            kappa,
            *args, **kwargs):
        return getattr(
            self, '_majorant', self._kernel
        )(
            t, *args, **kwargs
        ) * kappa.reshape(1, -1)

    def call_each(
            self,
            t,
            kappa,
            *args, **kwargs):
        return self._kernel(
            t, *args, **kwargs
        ) * kappa.reshape(1, -1)

    def integrate_each(
            self,
            t,
            kappa,
            *args, **kwargs):
        return self._integrate(
            t, *args, **kwargs
        ) * kappa.reshape(1, -1)


class ExpKernel(InfluenceKernel):
    def _kernel(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        theta = 1.0 / tau
        return theta * np.exp(-t * theta) * (t >= 0)

    def _integrate(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        theta = 1.0 / tau
        return 1 - np.exp(-t * theta) * (t >= 0)


class MaxwellKernel(InfluenceKernel):
    """
    http://mathworld.wolfram.com/MaxwellDistribution.html
    I think I could just use ``scipy.stats.maxwell``?
    That seems not to be autograd differentiable.
    """
    def _kernel(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        t2 = np.square(t)
        return np.sqrt(2.0/np.pi) * t2 * np.exp(
            -t2 / (2 * tau**2)
        )/(tau**3)

    def _mode(self, tau, *args, **kwargs):
        tau = np.asarray(tau).reshape(-1, 1)
        return np.sqrt(2) * tau

    def _integrate(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        return sp.special.erf(
            t / (np.sqrt(2)*tau)
        ) - t * np.sqrt(2.0/np.pi) / tau * np.exp(
            -np.square(t)/(2 * np.square(tau))
        )

    def _majorant(self, t, *args, **kwargs):
        mode = self._mode(*args, **kwargs)
        peak = self(mode, *args, **kwargs)
        print('umk', mode, peak)
        return np.choose(
            t > mode,
            [
                peak,
                self(t, *args, **kwargs)
            ]
        )


class GenericKernel(InfluenceKernel):
    """
    Construct a kernel from some functions
    """
    def __init__(
            self,
            kernel,
            majorant=None,
            integral=None,
            *args,
            **fixed_args
            ):
        self._kernel = kernel
        self._majorant = majorant if majorant is not None else kernel
        self._integral = integral
        super(GenericKernel, self).__init__(*args, **fixed_args)
