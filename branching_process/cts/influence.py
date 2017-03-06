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

    def __init__(self, *args, **fixed_args):
        self._fixed_args = fixed_args
        super(InfluenceKernel, self).__init__(*args)

    def majorant(self, t, *args, **kwargs):
        return np.squeeze(
            getattr(
                self, '_majorant', self._call
            )(t, *args, **kwargs, **self._fixed_args)
        )

    def __call__(self, t, *args, **kwargs):
        return np.squeeze(
            self._call(t, *args, **kwargs, **self._fixed_args)
        )

    def integrate(self, t, *args, **kwargs):
        # some kind of numerical quadrature fallback?
        return np.squeeze(
            self._integrate(t, *args, **kwargs, **self._fixed_args)
        )


class UniModalInfluenceKernel(InfluenceKernel):
    def mode(self, *args, **kwargs):
        return self._mode(**kwargs, **self._fixed_args)

    def _mode(self, *args, **kwargs):
        return 0

    def _majorant(self, t, *args, **kwargs):
        mode = self.mode(*args, **kwargs)
        peak = self(mode, *args, **kwargs)
        print('umk', mode, peak)
        return np.choose(
            t > mode,
            [
                peak,
                self(t, *args, **kwargs)
            ]
        )


class ExpKernel(InfluenceKernel):
    def _call(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        theta = 1.0 / tau
        return theta * np.exp(-t * theta) * (t >= 0)

    def _integrate(self, t, tau, *args, **kwargs):
        t = np.asarray(t).reshape(1, -1)
        tau = np.asarray(tau).reshape(-1, 1)
        theta = 1.0 / tau
        return 1 - np.exp(-t * theta) * (t >= 0)


class MaxwellKernel(UniModalInfluenceKernel):
    """
    http://mathworld.wolfram.com/MaxwellDistribution.html
    I think I could just use ``scipy.stats.maxwell``?
    That seems not to be autograd differentiable.
    """
    def _call(self, t, tau, *args, **kwargs):
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


class MultiKernel(InfluenceKernel):
    def __init__(
            self,
            n_kernels=1,
            kernel=MaxwellKernel(),
            *args, **fixed_args):
        self.kernel = kernel
        super(MultiKernel, self).__init__(*args, **fixed_args)

    def majorant(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.majorant_each(t, *args, **kwargs),
            0
        )

    def __call__(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.call_each(t, *args, **kwargs),
            0
        )

    def integrate(
            self,
            t,
            *args, **kwargs):
        return np.sum(
            self.integrate_each(t, *args, **kwargs),
            0
        )

    def majorant_each(
            self,
            t,
            *args, **kwargs):
        return self._majorant(
            t, *args, **kwargs)

    def call_each(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        return self._call(
            t, *args, **kwargs)

    def integrate_each(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        return self._integrate(
            t, *args, **kwargs)


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
