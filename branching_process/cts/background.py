"""
Background kernels are *like* InfluenceKernel, but not necessarily integrable
and there is no support for differentiationg with respect to time.
"""

from .influence import InfluenceKernel
have_autograd = False

try:
    import autograd.numpy as np
    have_autograd = True
except ImportError as e:
    import numpy as np


class ConstKernel(InfluenceKernel):
    def __init__(
            self,
            n_bases=0,
            kappa=None,
            eps=1e-8,
            *args, **fixed_kwargs):
        self._fixed_kwargs = fixed_kwargs
        if kappa is None and n_bases > 0:
            kappa = np.zeros(n_bases)
        if kappa is not None:
            self._fixed_kwargs.setdefault(
                'kappa', kappa
            )
        self.eps = eps
        self.n_bases = n_bases
        # super(ConstKernel, self).__init__(*args)

    def mu_bounds(self):
        return [(self.eps, None)]

    def kappa_bounds(self):
        return [(None, None)] * self.n_bases

    def f_kappa(self, **kwargs):
        mu = self.get_param('mu', 0.0, **kwargs)
        return np.maximum(mu, self.eps)

    def guess_params(self, **kwargs):
        # from IPython.core.debugger import Tracer; Tracer()()
        return self.guess_params_intensity(self.f_kappa(**kwargs))

    def guess_params_intensity(self, f_kappa_hat):
        med = np.mean(f_kappa_hat)
        return dict(
            mu=med
        )

    def __call__(self, t, *args, **kwargs):
        mu = self.get_params(**kwargs)['mu']
        return np.ones_like(t) * mu

    def integrate(self, t, *args, **kwargs):
        mu = self.get_params(**kwargs)['mu']
        return t * mu


class StepKernel(ConstKernel):
    """
    Piecewise-constant rate.
    This is presumably for background rate modelling.
    """
    def __init__(
            self,
            t_end=None,
            n_bases=None,
            *args,
            **fixed_kwargs
            ):
        if t_end is None:
            t_end = fixed_kwargs.get('tau', [0, 100])[-1]
        self.t_end = t_end
        if n_bases is None:
            if fixed_kwargs.get('tau', None) is not None:
                n_bases = np.asarray(fixed_kwargs.get('tau')).size - 1
            elif fixed_kwargs.get('kappa', None) is not None:
                n_bases = np.asarray(fixed_kwargs.get('kappa')).size
            else:
                n_bases = 100
        self.n_bases = n_bases
        fixed_kwargs.setdefault(
            'tau',
            np.linspace(0, t_end, n_bases+1, endpoint=True)
        )
        super(StepKernel, self).__init__(
            n_bases=n_bases,
            *args, **fixed_kwargs)

    def f_kappa(self, **kwargs):
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        return np.maximum(kappa + mu, self.eps)

    def guess_params_intensity(self, f_kappa_hat):
        med = np.median(f_kappa_hat)
        return dict(
            mu=med,
            kappa=f_kappa_hat-med
        )

    def __call__(self, t, *args, **kwargs):
        """
        """
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        f_kappa = self.f_kappa(kappa=kappa, mu=mu)
        tt = np.reshape(t, (-1, 1))
        stepwise_mask = (
            (tt >= tau[:-1].reshape(1, -1)) *
            (tt < tau[1:].reshape(1, -1))
        )
        outside = (t < tau[0]) + (t >= tau[-1])
        # from IPython.core.debugger import Tracer; Tracer()()
        return np.sum(
            stepwise_mask * np.reshape(f_kappa, (1, -1)),
            1
        ) + outside * mu

    def integrate(self, t, *args, **kwargs):
        """
        This integral is a simple linear interpolant,
        which I would like to do as a spline.
        However, I need to do it manually, since
        it needs to be autograd differentiable, which splines are not.
        The method here is not especially efficent.
        """
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        f_kappa = self.f_kappa(kappa=kappa, mu=mu)
        t = np.reshape(t, (-1, 1))
        delta = np.diff(tau)
        each = np.maximum(
            0, (t - tau[:-1].reshape(1, -1))
        )
        each = np.minimum(
            each,
            delta.reshape(1, -1)
        )
        return np.sum(
            each * np.reshape(f_kappa, (1, -1)),
            1
        ) + (mu * t.ravel())

    def majorant(self, t, *args, **kwargs):
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        kappa = np.maximum(kappa, -mu)
        return np.ones_like(t) * (mu + np.amax(kappa))

    def count(self, t, *args, **kwargs):
        tau = self.get_param('tau', **kwargs)
        return np.histogram(t, tau, density=False)


class MultiplicativeStepKernel(StepKernel):
    """
    Piecewise-constant rate.
    This is presumably for background rate modelling.
    """
    # def kappa_bounds(self):
    #     return [(-1, None)] * self.n_bases

    def f_kappa(self, **kwargs):
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        return (np.maximum(kappa + 1, self.eps)) * mu

    def guess_params_intensity(self, f_kappa_hat):
        # Is this correct?
        med = np.median(f_kappa_hat)
        return dict(
            mu=med,
            kappa=f_kappa_hat/med - 1
        )


class LogStepKernel(StepKernel):
    """
    Piecewise-constant rate.
    This is presumably for background rate modelling.
    """
    def kappa_bounds(self):
        return [(None, None)] * self.n_bases

    def f_kappa(self, **kwargs):
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        return mu * np.exp(kappa)

    def guess_params_intensity(self, f_kappa_hat):
        # Is this correct?
        med = np.median(f_kappa_hat)
        return dict(
            mu=med,
            kappa=np.log(f_kappa_hat/med)
        )


def as_background_kernel(
        function,
        majorant=None,
        integral=None,
        n_bases=0,
        t_start=0,
        t_end=100,
        **kwargs
        ):
    if hasattr(function, 'majorant'):
        return function
    elif function is None:
        # a number or None?
        return ConstKernel(
            **kwargs
        )
    elif n_bases == 0:
        # a number or None?
        return ConstKernel(
            mu=function,
            **kwargs
        )
    else:
        return StepKernel(
            mu=function,
            t_start=t_start,
            t_end=t_end,
            n_bases=n_bases,
            **kwargs
        )
