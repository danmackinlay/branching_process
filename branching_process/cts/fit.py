# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Poisson point process penalised likelihood regression.
"""

try:
    import autograd
    import autograd.numpy as np
    # import autograd.scipy as sp
    have_autograd = True
except ImportError as e:
    import numpy as np
    # import scipy as sp
    have_autograd = False

from scipy.optimize import minimize

from . import influence
from . import background
from . import model


class ContinuousExact(object):
    def __init__(
            self,
            debug=False
            ):
        self.debug = debug

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)
            print()

    def _debug_tee(self, label, obj):
        if self.debug:
            print(label, obj)
            print()
        return obj

    def _pack(
            self,
            mu=None,
            kappa=None,
            tau=None,
            omega=None,
            **kwargs
            ):
        """
        pack all params into one vector for blackbox optimisation;
        """
        mu = np.array(mu if mu is not None else self.params['mu']).ravel()
        pre_packed = [mu]
        if self._fit_kappa:
            pre_packed.append((
                kappa if kappa is not None else self.params['kappa']
            ).ravel())
        if self._fit_omega:
            pre_packed.append((
                omega if omega is not None else self.params['omega']
            ).ravel())
        if self._fit_tau:
            pre_packed.append((
                tau if tau is not None else self.params['tau']
            ).ravel())
        return np.concatenate(pre_packed)
        # self._debug_print('pk {!r}'.format(res))

    def _unpack(
            self,
            packed):
        """
        unpack all params from one vector for blackbox optimisation;
        """
        unpacked = dict(
            mu=packed[0],
        )
        if self._n_kappa_pack > 0:
            unpacked['kappa'] = packed[
                1:
                self._n_kappa_pack+1
            ]
        if self._n_omega_pack > 0:
            unpacked['omega'] = packed[
                self._n_kappa_pack + 1:
                self._n_kappa_pack + self._n_omega_pack + 1
            ]
        if self._n_tau_pack > 0:
            unpacked['tau'] = packed[
                self._n_kappa_pack + self._n_omega_pack + 1
            ]
        return unpacked

    def negloglik(
            self,
            **kwargs):

        return -model.loglik(
            ts=self._ts,
            eval_ts=self._eval_ts,
            phi_kernel=self.phi_kernel,
            mu_kernel=self.mu_kernel,
            t_start=self._t_start,
            t_end=self._t_end,
            **kwargs
        )

    def penalty(
            self,
            kappa=None,
            omega=None,
            pi_kappa=0.0,
            pi_omega=0.0,
            **kwargs
            ):
        if kappa is None:
            kappa = self.phi_kernel.get_param('kappa')
        if omega is None:
            omega = self.mu_kernel.get_param('kappa')

        # 2 different l_1 penalties
        pen = 0
        if self._fit_kappa:
            pen = pen + np.sum(
                np.abs(kappa) * pi_kappa
            )
        if self._fit_omega:
            pen = pen + np.sum(
                np.abs(omega) * pi_omega * self._omega_weight
            )
        return pen

    def dof(self,
            mu=None,
            kappa=None,
            tau=None,
            omega=None,
            **kwargs):
        if kappa is None:
            kappa = self.phi_kernel.get_param('kappa')
        if omega is None:
            omega = self.mu_kernel.get_param('kappa')
        if tau is None:
            tau = self.phi_kernel.get_param('tau')
        dof = 1  # mu
        if self._fit_tau:
            dof = dof + tau.size
        if self._fit_kappa:
            dof = dof + np.sum(
                np.abs(kappa) > self.tol
            )
        if self._fit_omega:
            dof = dof + np.sum(
                np.abs(omega) > self.tol
            )
        return dof

    def aic(self, method='aicc', negloglik=None, **kwargs):
        method = str.lower(method)
        dof = self.dof(**kwargs)
        if negloglik is None:
            negloglik = self.negloglik(**kwargs)
        if method == 'aic':
            return 2.0 * dof + 2.0 * negloglik
        elif method == 'aicc':
            return (
                2.0 * dof + 2.0 * negloglik
            ) + 2.0 * dof * (dof + 1) / max(self._n_ts-dof-1, 1)
        elif method == 'bic':
            return dof * np.log(self._n_ts) + 2.0 * negloglik

    def _setup_graphs(
            self,
            ts,
            phi_kernel=None,
            mu_kernel=None,
            n_phi_bases=1,
            n_mu_bases=0,
            fit_tau=False,
            fit_omega=False,
            fit_kappa=True,
            t_start=0.0,
            t_end=None,
            param_vector=None,
            tol=None,
            weight_count=False,
            **kwargs):

        self._t_start = t_start
        self._t_end = t_end or ts[-1]
        if tol is None:
            tol = 1e-5 * ts.size / (self._t_end - self._t_start)
        self.tol = tol
        self._ts = ts
        self._n_ts = ts[
            np.logical_and(
                ts >= self._t_start,
                ts < self._t_end
            )
        ].size
        # Full data likelihood must evaluate at the t_end also
        if ts[-1] < self._t_end:
            _eval_ts = np.append(
                ts[ts > self._t_start],
                [self._t_end]
            )
        else:
            _eval_ts = ts[
                np.logical_and(
                    ts >= self._t_start,
                    ts < self._t_end
                )
            ]

        self._eval_ts = _eval_ts

        if phi_kernel is None:
            phi_kernel = influence.MaxwellKernel(
                n_bases=n_phi_bases
            )
        else:
            phi_kernel = influence.as_influence_kernel(
                phi_kernel, n_bases=n_phi_bases
            )
        self.phi_kernel = phi_kernel
        self.n_phi_bases = phi_kernel.n_bases
        mu_kernel = background.as_background_kernel(
            mu_kernel,
            n_bases=n_mu_bases,
            t_start=self._t_start,
            t_end=self._t_end,
        )

        self.mu_kernel = mu_kernel
        self.n_mu_bases = mu_kernel.n_bases
        self._fit_tau = fit_tau if self.n_mu_bases > 0 else False
        self._fit_omega = fit_omega if self.n_mu_bases > 0 else False
        self._fit_kappa = fit_kappa
        self._n_kappa_pack = self.n_phi_bases * self._fit_kappa
        self._n_omega_pack = self.n_mu_bases * self._fit_omega
        self._n_tau_pack = self.n_phi_bases * self._fit_tau
        self._weight_count = weight_count
        self._mu_bounds = mu_kernel.mu_bounds()
        if self._fit_kappa:
            self._kappa_bounds = phi_kernel.kappa_bounds()
        else:
            self._kappa_bounds = []
        if self._fit_omega:
            self._omega_bounds = mu_kernel.kappa_bounds()
        else:
            self._omega_bounds = []
        if self._fit_tau:
            self._tau_bounds = phi_kernel.tau_bounds()
        else:
            self._tau_bounds = []
        self._all_bounds = (
            self._mu_bounds +
            self._kappa_bounds +
            self._omega_bounds +
            self._tau_bounds
        )
        if weight_count is None:
            weight_count = self._weight_count

        self._omega_weight = np.ones(self.n_mu_bases)

        # NB this will break if we ever fit the background rate tau
        if weight_count:
            self._omega_weight += np.array(self.mu_kernel.count(
                self._eval_ts,
                **kwargs
            ), dtype='float')

    def _teardown_graphs(self):

        del(self._ts)
        del(self._eval_ts)

    def fit(
            self,
            ts,
            phi_kernel=None,
            mu_kernel=None,
            n_phi_bases=1,
            n_mu_bases=0,
            fit_kappa=True,
            fit_tau=False,
            fit_omega=False,
            t_start=0.0,
            t_end=None,
            tol=None,
            coordwise=False,
            weight_count=False,
            **kwargs
            ):
        self._setup_graphs(
            ts=ts,
            phi_kernel=phi_kernel,
            mu_kernel=mu_kernel,
            n_phi_bases=n_phi_bases,
            n_mu_bases=n_mu_bases,
            fit_tau=fit_tau,
            fit_omega=fit_omega,
            fit_kappa=fit_kappa,
            t_start=t_start,
            t_end=t_end,
            param_vector=None,
            tol=tol,
            **kwargs
        )
        if coordwise:
            return self._fit_coordwise(
                **kwargs
            )
        else:
            return self._fit_simultaneous(
                **kwargs
            )

    def _guess_params(
            self,
            randomize=True,
            **guess
            ):
        """
        guess initialization params.
        Right now this inspects the kernels for defaults.

        This needs to be done better,
        since it is too easy to leak the true values into the fit
        from the simulation and to thus get
        unnaturally good results.

        TODO: mitigate this risk by not even requiring params unnecessarily
        """
        # We'd like to choose mu=0 as a guess
        # but this doesn't work for multiplicative background noise
        # So we choose a  background intensity of the correct order
        # of magnitude, so that kappa is not negative for the first round.
        if 'mu' not in guess:
            guess['mu'] = self._n_ts/(self._t_end - self._t_start) * 0.9
        if randomize:
            guess['mu'] *= np.random.uniform(size=1)
        if self._fit_kappa:
            if 'kappa' not in guess:
                kappa = np.ones(self.phi_kernel.n_bases)
                kappa /= kappa.size
                if randomize:
                    kappa *= np.random.uniform(high=2, size=kappa.shape)
                guess['kappa'] = kappa
        if self._fit_omega:
            if 'omega' not in guess:
                if randomize:
                    omega = np.random.normal(size=self.mu_kernel.n_bases)
                else:
                    omega = np.zeros(self.mu_kernel.n_bases)
                guess['omega'] = omega
        if self._fit_tau:
            if 'tau' not in guess:
                tau = np.ones_like(self.phi_kernel.n_bases)
                tau /= tau.size
                if randomize:
                    tau *= np.random.normal(size=tau.shape)
                guess['tau'] = tau
        return guess

    def objective(
            self,
            **kwargs):
        loss_negloglik = self.negloglik(
            **kwargs
        )
        loss_penalty = self.penalty(
            **kwargs
        )
        # self._debug_print('insobj', loss_negloglik, loss_penalty)
        return (loss_negloglik + loss_penalty)

    def obj_packed(self, x, other_params={}):
        kwargs = dict(other_params)
        kwargs.update(self._unpack(x))
        return self.objective(**kwargs)

    def obj_mu(self, x, other_params={}):
        other_kwargs = dict(self.params)
        other_kwargs.update(other_params)
        other_kwargs['mu'] = x
        return self.objective(**other_kwargs)

    def obj_kappa(self, x, other_params={}):
        other_kwargs = dict(self.params)
        other_kwargs.update(other_params)
        other_kwargs['kappa'] = x
        return self.objective(**other_kwargs)

    def obj_tau(self, x, other_params={}):
        other_kwargs = dict(self.params)
        other_kwargs.update(other_params)
        other_kwargs['tau'] = x
        return self.objective(**other_kwargs)

    def obj_omega(self, x, other_params={}):
        other_kwargs = dict(self.params)
        other_kwargs.update(other_params)
        other_kwargs['omega'] = x
        return self.objective(**other_kwargs)

    def _setup_grad(self):
        self._grad_mu = autograd.grad(
            self.obj_mu, 0)
        self._grad_kappa = autograd.grad(
            self.obj_kappa, 0)
        self._grad_tau = autograd.grad(
            self.obj_tau, 0)
        self._grad_omega = autograd.grad(
            self.obj_omega, 0)
        self._grad_packed = autograd.grad(
            self.obj_packed, 0)
        # self.___grad_packed = autograd.grad(
        #     self.obj_packed, 0)
        #
        # def _(x, *args, **kwargs):
        #     print('xg', x)
        #     g = self.___grad_packed(x, *args, **kwargs)
        #     print('gg', g)
        #     return g
        # self._grad_packed = _

    def _fit_coordwise(
            self,
            max_steps=10,
            step_iter=25,
            eps=1e-8,
            warm=False,
            method='TNC',  # or 'L-BFGS-B'
            **kwargs
            ):
        """
        fit by truncated Newton in each coordinate group
        """
        if not warm:
            self.params = self._guess_params(**kwargs)
        self._setup_grad()

        # self._hessian_diag_negloglik = autograd.elementwise_grad(
        #     self._grad_negloglik, 0)

        fit = dict(**self.params)

        for i in range(max_steps):
            self._debug_print('fit', fit)
            new_fit = {}
            res = minimize(
                self.obj_kappa,
                x0=fit['kappa'],
                args=(new_fit,),
                method=method,
                # method='L-BFGS-B',
                jac=self._grad_kappa,
                bounds=self._kappa_bounds,
                callback=lambda x: self._debug_tee('kappa_fit', x),
                options=dict(
                    maxiter=step_iter,
                    disp=self.debug
                )
            )
            kappa = res.x
            # from IPython.core.debugger import Tracer; Tracer()()
            kappa[np.abs(kappa < self.tol)] = 0
            new_fit['kappa'] = kappa
            self._debug_print('new_fit', new_fit)

            if self._fit_tau:
                res = minimize(
                    self.obj_tau,
                    x0=fit['tau'],
                    args=(new_fit,),
                    method='L-BFGS-B',  # ?
                    jac=self._grad_tau,
                    bounds=self._tau_bounds,
                    callback=lambda x: self._debug_tee('tau_fit', x),
                    options=dict(
                        maxiter=step_iter,
                        disp=self.debug
                    )
                )
                tau = res.x
                new_fit['tau'] = tau
                self._debug_print('new_fit', new_fit)

            if self._fit_omega:
                res = minimize(
                    self.obj_omega,
                    x0=fit['omega'],
                    args=(new_fit,),
                    method=method,
                    jac=self._grad_omega,
                    bounds=self._omega_bounds,
                    callback=lambda x: self._debug_tee('omega_fit', x),
                    options=dict(
                        maxiter=step_iter,
                        disp=self.debug
                    )
                )
                omega = res.x
                omega[np.abs(omega < self.tol)] = 0
                new_fit['omega'] = omega
                self._debug_print('new_fit', new_fit)

            # one-step mu update is possible for known noise structure
            # e.g. additive, but not in general. So let's not.
            res = minimize(
                self.obj_mu,
                x0=fit['mu'],
                args=(new_fit,),
                method=method,
                jac=self._grad_mu,
                bounds=self._mu_bounds,
                callback=lambda x: self._debug_tee('mu_fit', x),
                options=dict(
                    maxiter=step_iter,
                    disp=self.debug
                )
            )
            mu = res.x
            new_fit['mu'] = mu
            self._debug_print('new_fit', new_fit)

            fit.update(new_fit)
            self.params.update(fit)

        self.params['negloglik'] = self.negloglik(**self.params)
        self.params['dof'] = self.dof(**self.params)
        self.params['aic'] = self.aic(method='aic', **self.params)
        self.params['bic'] = self.aic(method='bic', **self.params)
        self.params['aicc'] = self.aic(method='aicc', **self.params)
        return self.params

    def _fit_simultaneous(
            self,
            step_iter=100,
            eps=1e-8,
            warm=False,
            method='TNC',  # or 'L-BFGS-B'
            refit_mu=True,
            **kwargs
            ):
        """
        fit by truncated Newton in all coordinates.
        """
        if not warm:
            self.params = self._guess_params(**kwargs)
        self._setup_grad()

        # self._hessian_diag_negloglik = autograd.elementwise_grad(
        #     self._grad_negloglik, 0)

        fit = dict(**self.params)
        x0 = self._pack(**fit)
        self._debug_print('fit {!r}/{}'.format(fit, x0))

        res = minimize(
            self.obj_packed,
            x0=x0,
            args=(self.params,),
            method=method,
            jac=self._grad_packed,
            bounds=self._all_bounds,
            callback=lambda x: self._debug_tee('packed_fit', x),
            options=dict(
                maxiter=step_iter,
                disp=self.debug
            )
        )
        new_fit = self._unpack(res.x)
        jac = res.jac

        # We can improve convergence by taking
        # analytic shortcuts for wide, low
        # mu kernel params
        if refit_mu:
            if not hasattr(self.mu_kernel, 'guess_params'):
                # brute force version:
                res = minimize(
                    self.obj_mu,
                    x0=fit['mu'],
                    args=(new_fit,),
                    method=method,
                    jac=self._grad_mu,
                    bounds=self._mu_bounds,
                    callback=lambda x: self._debug_tee('mu_fit', x),
                    options=dict(
                        maxiter=step_iter,
                        disp=self.debug
                    )
                )
                mu = res.x
                new_fit['mu'] = mu
                self._debug_print('new_fit', new_fit)
                jac[0] = res.jac
            else:
                guess_kwargs = dict(mu=new_fit['mu'])
                if self._fit_omega:
                    guess_kwargs['kappa'] = new_fit['omega']
                partial_fit = self._debug_tee(
                    'one_step_mu',
                    self.mu_kernel.guess_params(**guess_kwargs)
                )
                if self._fit_omega:
                    new_fit['omega'] = partial_fit['kappa']
                new_fit['mu'] = partial_fit['mu']

        if self._fit_kappa:
            new_fit['kappa'][np.abs(new_fit['kappa'] < self.tol)] = 0
        if self._fit_omega:
            new_fit['omega'][np.abs(new_fit['omega'] < self.tol)] = 0
        fit.update(new_fit)

        self.params.update(fit)
        self.params['negloglik'] = self.negloglik(**self.params)
        self.params['dof'] = self.dof(**self.params)
        self.params['aic'] = self.aic(method='aic', **self.params)
        self.params['bic'] = self.aic(method='bic', **self.params)
        self.params['aicc'] = self.aic(method='aicc', **self.params)
        self.params['jac'] = jac  # for diagnosis
        self.params['penalty'] = self.penalty(**self.params)  # for diagnosis
        return self.params


class FitHistory(object):
    def __init__(self, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._keys = kwargs.keys()

    def __repr__(self):
        return 'FitHistory({param!r})'.format(param=self.param)
