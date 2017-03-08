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
        super(InfluenceKernel, self).__init__(*args)

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
        t = np.asarray(t)
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        tau = np.asarray(new_kwargs.pop('tau'))
        kappa = np.asarray(new_kwargs.pop('kappa'))
        return getattr(
            self, '_majorant', self._kernel
        )(
            t=t.reshape(-1, 1),
            tau=tau.reshape(1, -1),
            *args, **new_kwargs
        ) * kappa.reshape(1, -1)

    def call_each(
            self,
            t,
            *args, **kwargs):
        t = np.asarray(t)
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        tau = np.asarray(new_kwargs.pop('tau'))
        kappa = np.asarray(new_kwargs.pop('kappa'))

        return self._kernel(
            t=t.reshape(-1, 1),
            tau=tau.reshape(1, -1),
            *args, **new_kwargs
        ) * kappa.reshape(1, -1)

    def integrate_each(
            self,
            t,
            *args, **kwargs):
        t = np.asarray(t)
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        tau = np.asarray(new_kwargs.pop('tau'))
        kappa = np.asarray(new_kwargs.pop('kappa'))
        return self._integrate(
            t=t.reshape(-1, 1),
            tau=tau.reshape(1, -1),
            *args, **new_kwargs
        ) * kappa.reshape(1, -1)


class ExpKernel(InfluenceKernel):
    def _kernel(self, t, tau, *args, **kwargs):
        t = np.asarray(t)
        tau = np.asarray(tau)
        theta = 1.0 / tau
        return theta * np.exp(-t * theta) * (t >= 0)

    def _integrate(self, t, tau, *args, **kwargs):
        t = np.asarray(t)
        tau = np.asarray(tau)
        theta = 1.0 / tau
        return 1 - np.exp(-t * theta) * (t >= 0)


class MaxwellKernel(InfluenceKernel):
    """
    http://mathworld.wolfram.com/MaxwellDistribution.html
    I think I could just use ``scipy.stats.maxwell``?
    That seems not to be autograd differentiable.
    """
    def _kernel(self, t, tau, *args, **kwargs):
        t = np.asarray(t)
        tau = np.asarray(tau)
        t2 = np.square(t)
        return np.sqrt(2.0/np.pi) * t2 * np.exp(
            -t2 / (2 * tau**2)
        )/(tau**3)

    def _integrate(self, t, tau, *args, **kwargs):
        t = np.asarray(t)
        tau = np.asarray(tau)
        return sp.special.erf(
            t / (np.sqrt(2)*tau)
        ) - t * np.sqrt(2.0/np.pi) / tau * np.exp(
            -np.square(t)/(2 * np.square(tau))
        )

    def _majorant(self, t, tau, *args, **kwargs):
        tau = np.asarray(tau)
        t = np.asarray(t)
        mode = np.sqrt(2) * tau
        peak = self._kernel(mode, tau=tau, *args, **kwargs)
        return np.choose(
            t > mode,
            [
                peak,
                self._kernel(t, tau=tau, *args, **kwargs)
            ]
        )


class StepKernel(InfluenceKernel):
    """
    Piecewise-constant rate.
    This is presumed to be for background rate modelling.
    """
    def __init__(
            self,
            end,
            n_bases=100,
            *args,
            **fixed_args
            ):
        self.end = end
        super(StepKernel, self).__init__(n_bases=n_bases, *args, **fixed_args)
        self._fixed_args.setdefault(
            'tau',
            np.linspace(0, end, n_bases+1, endpoint=True)
        )

    def __call__(self, t, *args, **kwargs):
        """
        because we need n_bases+1 tau points for this, it doesn't map onto
        the easy structure of the others.
        I'm not convinced this subclassing rigmarole is worth the effort.
        """
        t = np.asarray(t)
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        tau = np.asarray(new_kwargs.pop('tau'))
        mu = new_kwargs.pop('mu', 0.0)
        kappa = np.asarray(new_kwargs.pop('kappa'))
        kappa = np.maximum(kappa, -mu)
        t = t.reshape(-1, 1)
        each = (
            (t > tau[:-1].reshape(1, -1)) -
            (t > tau[1:].reshape(1, -1))
        )
        return np.sum(
            each * kappa.reshape(1, -1),
            1
        ) + mu

    def integrate(self, t, *args, **kwargs):
        t = np.asarray(t)
        new_kwargs = dict()
        new_kwargs.update(self._fixed_args, **kwargs)
        tau = np.asarray(new_kwargs.pop('tau'))
        mu = new_kwargs.pop('mu', 0.0)
        kappa = np.asarray(new_kwargs.pop('kappa'))
        kappa = np.maximum(kappa, -mu)
        t = t.reshape(-1, 1)
        delta = np.diff(tau)
        each = np.maximum(
            0, (t - tau[:-1].reshape(1, -1))
        )
        each = np.minimum(
            each,
            delta.reshape(1, -1)
        )
        return np.sum(
            each * kappa.reshape(1, -1),
            1
        ) + (mu * t.ravel())


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


def as_influence_kernel(
        function,
        majorant=None,
        integral=None
    ):
    if hasattr(function, 'majorant'):
        return function
    else:
        return GenericKernel(
            kernel=function,
            majorant=majorant,
            integral=integral
        )
