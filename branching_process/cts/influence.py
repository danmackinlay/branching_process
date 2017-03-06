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

    def majorant(self, t, tau, *args, **kwargs):
        return self(t, tau=tau, *args, **kwargs)

    def __call__(self, t, tau, *args, **kwargs):
        raise NotImplementedError()

    def integrate(self, t, tau, *args, **kwargs):
        # some kind of quadrature?
        raise NotImplementedError()


class UniModalInfluenceKernel(InfluenceKernel):
    def mode(self, tau, *args, **kwargs):
        return 0

    def majorant(self, t, tau, *args, **kwargs):
        mode = self.mode(tau)
        peak = self(mode, tau)
        return np.choose(
            t > mode,
            [
                peak,
                self(t, tau, *args, **kwargs)
            ]
        )


class ExpKernel(InfluenceKernel):
    def __call__(self, t, tau, *args, **kwargs):
        theta = 1.0 / tau
        return theta * np.exp(-t * theta) * (t >= 0)

    def integrate(self, t, tau, *args, **kwargs):
        theta = 1.0 / tau
        return 1-np.exp(-t * theta) * (t >= 0)


class MaxwellKernel(UniModalInfluenceKernel):
    """
    http://mathworld.wolfram.com/MaxwellDistribution.html
    I think I could just use ``scipy.stats.maxwell``?
    That seems not to be autograd differentiable.
    """
    def __call__(self, t, tau, *args, **kwargs):
        t2 = np.square(t)
        return np.sqrt(2.0/np.pi) * t2 * np.exp(
            -t2 / (2 * tau**2)
        )/(tau**3)

    def mode(self, tau, *args, **kwargs):
        return np.sqrt(2) * tau

    def integrate(self, t, tau, *args, **kwargs):
        return sp.special.erf(
            t / (np.sqrt(2)*tau)
        ) - t * np.sqrt(2.0/np.pi) / tau * np.exp(
            -np.square(t)/(2 * np.square(tau))
        )


class MultiKernel(object):
    def __init__(
            self,
            n_kernels=1,
            kernel=MaxwellKernel(),
            *args, **kwargs):
        self.kernel = kernel

    def majorant(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        return self.majorant(t, tau=tau, kappa=kappa, *args, **kwargs)

    def __call__(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        pass

    def integrate(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        pass

    def majorant_each(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        out = np.zeros(self.n_kernels)
        if np.isscalar(kappa):
            kappa = np.ones(self.n_kernels) * kappa
        for i in range(self.n_kernels):
            out[i] = self.kernel.majorant(t, tau=tau[i], kappa=kappa[i])
        return out

    def call_each(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        out = np.zeros(self.n_kernels)
        if np.isscalar(kappa):
            kappa = np.ones(self.n_kernels) * kappa
        for i in range(self.n_kernels):
            out[i] = self.kernel(t, tau=tau[i], kappa=kappa[i])
        return out

    def integrate_each(
            self,
            t,
            tau=None,
            kappa=None,
            *args, **kwargs):
        out = np.zeros(self.n_kernels)
        if np.isscalar(kappa):
            kappa = np.ones(self.n_kernels) * kappa
        for i in range(self.n_kernels):
            out[i] = self.kernel.integrate(t, tau=tau[i], kappa=kappa[i])
        return out


# replace with a wrapper function? yes.
class FixedKernel(InfluenceKernel):
    def __init__(
            self,
            kernel,
            tau=1.0,
            kappa=1.0,
            ):
        self.tau = tau
        self.kappa = kappa
        self.kernel = kernel

    def majorant(self, t, *args, **kwargs):
        return self.kernel.majorant(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)

    def __call__(self, t, *args, **kwargs):
        return self.kernel(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)

    def integrate(self, t, *args, **kwargs):
        return self.kernel.integrate(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)
