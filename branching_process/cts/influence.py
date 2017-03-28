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
            n_bases=1,
            *args, **fixed_args):
        self._fixed_args = fixed_args
        self._fixed_args.setdefault('kappa', np.ones(n_bases)/n_bases)
        self._fixed_args.setdefault('tau', np.arange(n_bases))
        self.n_bases = n_bases
        super(InfluenceKernel, self).__init__(*args)

    def get_param(self, key, fallback=None, **kwargs):
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        return new_kwargs.get(key, fallback)

    def majorant(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.majorant_each(t, *args, **kwargs),
            1
        )

    def __call__(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.call_each(t, *args, **kwargs),
            1
        )

    def integrate(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.integrate_each(t, *args, **kwargs),
            1
        )

    def majorant_each(
            self,
            t,
            *args, **kwargs):
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        return getattr(
            self, '_majorant', self._kernel
        )(
            t=np.reshape(t, (-1, 1)),
            tau=np.reshape(tau, (1, -1)),
            *args, **kwargs
        ) * np.reshape(kappa, (1, -1))

    def call_each(
            self,
            t,
            *args, **kwargs):
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)

        return self._kernel(
            t=np.reshape(t, (-1, 1)),
            tau=np.reshape(tau, (1, -1)),
            *args, **kwargs
        ) * np.reshape(kappa, (1, -1))

    def integrate_each(
            self,
            t,
            *args, **kwargs):
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        return self._integrate(
            t=np.reshape(t, (-1, 1)),
            tau=np.reshape(tau, (1, -1)),
            *args, **kwargs
        ) * np.reshape(kappa, (1, -1))


class ExpKernel(InfluenceKernel):
    def _kernel(self, t, tau, *args, **kwargs):
        theta = 1.0 / tau
        return theta * np.exp(-t * theta) * (t >= 0)

    def _integrate(self, t, tau, *args, **kwargs):
        theta = 1.0 / tau
        return 1 - np.exp(-t * theta) * (t >= 0)


class MaxwellKernel(InfluenceKernel):
    """
    http://mathworld.wolfram.com/MaxwellDistribution.html
    I think I could just use ``scipy.stats.maxwell``?
    That seems not to be autograd differentiable.
    """
    def _kernel(self, t, tau, *args, **kwargs):
        t2 = np.square(t)
        return np.sqrt(2.0/np.pi) * t2 * np.exp(
            -t2 / (2 * tau**2)
        )/(tau**3)

    def _integrate(self, t, tau, *args, **kwargs):
        return sp.special.erf(
            t / (np.sqrt(2)*tau)
        ) - t * np.sqrt(2.0/np.pi) / tau * np.exp(
            -np.square(t)/(2 * np.square(tau))
        )

    def _majorant(self, t, tau, *args, **kwargs):
        mode = np.sqrt(2) * tau
        peak = self._kernel(mode, tau=tau, *args, **kwargs)
        return np.choose(
            t > mode,
            [
                peak,
                self._kernel(t, tau=tau, *args, **kwargs)
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
        self._integrate = integral
        super(GenericKernel, self).__init__(*args, **fixed_args)


def as_influence_kernel(
        function,
        majorant=None,
        integral=None,
        n_bases=1,
        **kwargs
        ):
    if hasattr(function, 'majorant'):
        return function
    elif not callable(function):
        # a number or None?
        return ConstKernel(mu=function or 0.0)
    else:
        # a function, but not a kernel
        return GenericKernel(
            kernel=function,
            majorant=majorant,
            integral=integral,
            n_bases=n_bases
        )
