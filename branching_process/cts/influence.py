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
            eps=1e-8,
            *args, **fixed_kwargs):
        self._fixed_kwargs = fixed_kwargs
        self._fixed_kwargs.setdefault('kappa', np.ones(n_bases)/n_bases)
        self._fixed_kwargs.setdefault('tau', np.arange(n_bases)+1)
        self.n_bases = n_bases
        self.eps = eps
        super(InfluenceKernel, self).__init__(*args)

    def kappa_bounds(self):
        return [(0, 1)] * self.n_bases

    def tau_bounds(self):
        return [(self.eps, None)] * self.n_bases

    def get_param(self, key, fallback=None, **kwargs):
        return self.get_params(**kwargs).get(key, fallback)

    def get_params(self, **kwargs):
        new_kwargs = dict(**self._fixed_kwargs)
        for key, val in kwargs.items():
            if val is not None:
                new_kwargs[key] = val
        return new_kwargs

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
        params = self.get_params(**kwargs)
        params['tau'] = np.reshape(params['tau'], (1, -1))
        return getattr(
            self, '_majorant', self._kernel
        )(
            t=np.reshape(t, (-1, 1)),
            *args, **params
        ) * np.reshape(params['kappa'], (1, -1))

    def call_each(
            self,
            t,
            *args, **kwargs):
        params = self.get_params(**kwargs)
        params['tau'] = np.reshape(params['tau'], (1, -1))

        return self._kernel(
            t=np.reshape(t, (-1, 1)),
            *args, **params
        ) * np.reshape(params['kappa'], (1, -1))

    def integrate_each(
            self,
            t,
            *args, **kwargs):
        params = self.get_params(**kwargs)
        params['tau'] = np.reshape(params['tau'], (1, -1))
        return self._integrate(
            t=np.reshape(t, (-1, 1)),
            *args, **params
        ) * np.reshape(params['kappa'], (1, -1))


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
            **fixed_kwargs
            ):
        self._kernel = kernel
        self._majorant = majorant if majorant is not None else kernel
        self._integrate = integral
        super(GenericKernel, self).__init__(*args, **fixed_kwargs)


def as_influence_kernel(
        function=None,
        majorant=None,
        integral=None,
        n_bases=1,
        **kwargs
        ):
    if function is None:
        return ExpKernel(
            n_bases=n_bases,
            **kwargs
        )
    elif hasattr(function, 'majorant'):
        return function
    elif callable(function):
        # a function, but not yet a kernel
        return GenericKernel(
            kernel=function,
            majorant=majorant,
            integral=integral,
            n_bases=n_bases
        )
    else:
        raise ValueError(
            "How should I interpret {!r} as an influence kernel?".format(
                function
            )
        )
