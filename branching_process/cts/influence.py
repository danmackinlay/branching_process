try:
    import autograd
    import autograd.numpy as np
    from autograd.numpy import sqrt, pi
    have_autograd = True
except ImportError as e:
    import numpy as np
    have_autograd = False
    from np import sqrt, pi


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
        peak = self(mode)
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


class MaxwellKernel(UniModalInfluenceKernel):
    """
    I think I could just use ``scipy.stats.maxwell``?
    """
    def __call__(self, t, tau, *args, **kwargs):
        t2 = np.square(t)
        return sqrt(2.0/pi) * t2 * np.exp(
            -t2 / (2 * tau**2)
        )/(tau**3)

    def mode(self, tau, *args, **kwargs):
        return sqrt(2) * tau


class MultiKernel(object):
    def __init__(
            self,
            n_kernels=1,
            kernel=MaxwellKernel(),
            *args, **kwargs):
        self.kernel=kernel

    def majorant(self, t, tau=None, kappa=None, *args, **kwargs):
        return self(t, tau=tau, kappa=kappa, *args, **kwargs)

    def __call__(self, t, tau=None, kappa=None, *args, **kwargs):
        pass

    def integrate(self, t, tau=None, kappa=None, *args, **kwargs):
        pass

    def majorant_all(self, t, tau=None, kappa=None, *args, **kwargs):
        out = np.zeros(n_kernels)
        if np.isscalar(kappa):
            kappa = np.ones(self.n_kernels) * kappa
        for i in range(n_kernels):
            out[i] = kernel(t, tau=tau[i], kappa=kap
        return out

    def call_all(self, t, tau=None, kappa=None, *args, **kwargs):
        pass

    def integrate_all(self, t, tau=None, kappa=None, *args, **kwargs):
        pass


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

    def majorant(self, t, *args, **kwargs):
        return kernel.majorant(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)

    def __call__(self, t, *args, **kwargs):
        return kernel(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)

    def integrate(self, t, *args, **kwargs):
        return kernel.integrate(
            t, tau=self.tau, kappa=self.kappa,
            *args, **kwargs)
