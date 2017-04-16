"""
BackgroundKernel is *like* InfluenceKernel, but not necessarily integrable
and there is no support for differentiationg with respect to time.
"""

from .influence import InfluenceKernel
have_autograd = False

try:
    import autograd.numpy as np
    have_autograd = True
except ImportError as e:
    import numpy as np


class BackgroundKernel(InfluenceKernel):
    def __init__(
            self,
            n_bases=0,
            kappa=None,
            *args, **fixed_args):
        self._default_kwargs = fixed_args
        self._default_kwargs.setdefault(
            'kappa',
            kappa if kappa is not None else np.ones(n_bases))
        self.n_bases = n_bases
        # super(BackgroundKernel, self).__init__(*args)


class ConstKernel(BackgroundKernel):
    """
    Constant rate.
    This is presumably for background rate modelling.
    """
    def __init__(
            self,
            *args,
            **fixed_args
            ):
        super(ConstKernel, self).__init__(
            n_bases=0,
            *args, **fixed_args)

    def __call__(self, t, *args, **kwargs):
        mu = self.get_params(**kwargs)['mu']
        return np.ones_like(t) * mu

    def integrate(self, t, *args, **kwargs):
        mu = self.get_params(**kwargs)['mu']
        return t * mu


class AdditiveStepKernel(BackgroundKernel):
    """
    Piecewise-constant rate.
    This is presumably for background rate modelling.
    """
    def __init__(
            self,
            t_end=None,
            n_bases=None,
            *args,
            **fixed_args
            ):
        if t_end is None:
            t_end = fixed_args.get('tau', [0, 100])[-1]
        self.t_end = t_end
        if n_bases is None:
            if fixed_args.get('tau', None) is not None:
                n_bases = np.asarray(fixed_args.get('tau')).size - 1
            elif fixed_args.get('kappa', None) is not None:
                n_bases = np.asarray(fixed_args.get('kapp')).size
            else:
                n_bases = 100
        self.n_bases = n_bases
        fixed_args.setdefault(
            'tau',
            np.linspace(0, t_end, n_bases+1, endpoint=True)
        )
        super(AdditiveStepKernel, self).__init__(
            n_bases=n_bases,
            *args, **fixed_args)

    def omega(self, kappa, mu, **kwargs):
        return np.maximum(kappa + mu, 0)

    def __call__(self, t, *args, **kwargs):
        """
        """
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        omega = self.omega(kappa, mu)
        tt = np.reshape(t, (-1, 1))
        stepwise_mask = (
            (tt >= tau[:-1].reshape(1, -1)) *
            (tt < tau[1:].reshape(1, -1))
        )
        outside = (t < tau[0]) + (t >= tau[-1])
        # from IPython.core.debugger import Tracer; Tracer()()
        return np.sum(
            stepwise_mask * np.reshape(omega, (1, -1)),
            1
        ) + outside * mu

    def integrate(self, t, *args, **kwargs):
        """
        This is tedious to construct since
        it needs to be autograd differentiable, which splines are not.
        """
        tau = self.get_param('tau', **kwargs)
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        omega = self.omega(kappa, mu)
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
            each * np.reshape(omega, (1, -1)),
            1
        ) + (mu * t.ravel())

    def majorant(self, t, *args, **kwargs):
        kappa = self.get_param('kappa', **kwargs)
        mu = self.get_param('mu', 0.0, **kwargs)
        kappa = np.maximum(kappa, -mu)
        return np.ones_like(t) * (mu + np.amax(kappa))


class MultiplicativeStepKernel(AdditiveStepKernel):
    """
    Piecewise-constant rate.
    This is presumably for background rate modelling.
    """
    def omega(self, kappa, mu, **kwargs):
        return (1 + np.maximum(kappa, -1)) * mu


def as_background_kernel(
        function,
        majorant=None,
        integral=None,
        n_bases=1,
        **kwargs
        ):
    if hasattr(function, 'majorant'):
        return function
    else:
        # a number or None?
        return ConstKernel(mu=function or 0.0)
