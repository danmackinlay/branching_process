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
    from autograd.util import quick_grad_check, check_grads, nd, unary_nd
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

    def _debug_tee(self, label, obj):
        if self.debug:
            print(label, obj)
        return obj

    def _pack(
            self,
            mu=1.0,
            kappa=0.0,
            tau=None,
            omega=None
            ):
        """
        pack all params into one vector for blackbox optimisation;
        currently unused
        """

        packed = np.zeros(
            1 +
            self.n_phi_bases +
            self.n_tau +
            self.n_omega
        )
        packed[0] = mu
        packed[
            1:self.n_phi_bases + 1
        ] = kappa
        packed[
            self.n_phi_bases + 1:
            self.n_phi_bases + self.n_tau + 1
        ] = tau if tau is not None else 1.0
        packed[
            self.n_phi_bases + self.n_tau + 1:
        ] = omega if omega is not None else 0.0
        return packed

    def _unpack(
            self,
            packed):
        """
        unpack all params from one vector for blackbox optimisation;
        currently unused.
        """
        unpacked = dict(
            mu=packed[0],
            kappa=packed[1:self.n_phi_bases+1],
        )
        # are we fitting time parameters?
        if self.n_tau > 0:
            unpacked = packed[
                self.n_phi_bases + 1:
                self.n_phi_bases + self.n_tau + 1
            ]
        # omega parameters - exogenous variation
        if self.n_omega > 0:
            unpacked = packed[
                self.n_phi_bases + self.n_tau + 1:
            ]
        return unpacked

    def negloglik_loss(
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

    def _penalty_weight_packed(
            self,
            pi_kappa=0.0,
            pi_omega=1e-8,
            **kwargs
            ):
        return self._pack(
            mu=0.0,
            kappa=pi_kappa,
            omega=pi_omega
        )  # / self.penalty_scale

    def penalty(
            self,
            mu=1.0,
            kappa=0.0,
            tau=1.0,
            omega=0.0,
            pi_kappa=0.0,
            pi_omega=0.0,
            **kwargs
            ):
        # 2 different l_1 penalties
        return np.sum(
            np.abs(kappa) * pi_kappa
        ) + np.sum(
            np.abs(omega) * pi_omega
        )

    def _setup_graphs(
            self,
            ts,
            phi_kernel=None,
            mu_kernel=None,
            n_phi_bases=1,
            n_mu_bases=0,
            fit_tau=False,
            fit_omega=False,
            t_start=0.0,
            t_end=None,
            param_vector=None,
            **kwargs):

        self._t_start = t_start
        self._t_end = t_end or ts[-1]
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
            phi_kernel = influence.MaxwellKernel(n_bases=5)
        else:
            phi_kernel = influence.as_influence_kernel(
                phi_kernel, n_bases=n_phi_bases
            )
        self.phi_kernel = phi_kernel
        self.n_phi_bases = phi_kernel.n_bases
        if mu_kernel is None:
            if n_mu_bases > 0:
                mu_kernel = background.AdditiveStepKernel(
                    t_start=self._t_start,
                    t_end=self._t_end,
                    n_bases=n_mu_bases)
            else:
                mu_kernel = background.ConstKernel()
        self.mu_kernel = mu_kernel
        self.n_mu_bases = n_mu_bases

        self._fit_tau = fit_tau
        self._fit_omega = fit_omega
        self.n_tau = self.n_phi_bases * self._fit_tau
        self.n_omega = self.n_mu_bases * self._fit_omega
        self._mu_bounds = [(0, None)]
        self._kappa_bounds = [(0, 1)] * self.n_phi_bases
        self._tau_bounds = [(0, None)] * self.n_tau
        self._omega_bounds = [(0, None)] * self.n_omega

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
            fit_tau=False,
            fit_omega=False,
            t_start=0.0,
            t_end=None,
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
            **kwargs
        )
        return self._fit(
            **kwargs
        )

    def _guess_params(
            self,
            mu=None,
            kappa=None,
            tau=None,
            omega=None,
            pi_kappa=0.0,
            pi_omega=1e-8,
            **kwargs
            ):
        """
        fit by truncated Newton in each coordinate group
        """
        if tau is None:
            tau = self.phi_kernel.get_param('tau')
        if omega is None:
            omega = self.mu_kernel.get_param('kappa')
        if kappa is None:
            kappa = self.phi_kernel.get_param('kappa')
        if mu is None:
            # We'd like to choose mu=0 as a guess
            # but this doesn't work for multiplicative background noise
            # So we choose a low background intensity of the correct order
            # of magnitude, so that kappa is not negative for the first round.
            mu = self.mu_kernel.get_param(
                'mu',
                self._n_ts/(self._t_end - self._t_start) * 0.1
            )
        return dict(
            mu=mu,
            kappa=kappa,
            tau=tau,
            omega=omega,
            pi_omega=pi_omega,
            pi_kappa=pi_kappa,
        )

    def objective(
            self,
            **kwargs):
        loss_negloglik = self.negloglik_loss(
            **kwargs
        )
        loss_penalty = self.penalty(
            **kwargs
        )
        # self._debug_print('insobj', loss_negloglik, loss_penalty)
        return (loss_negloglik + loss_penalty)

    # def grad_objective(
    #         self,
    #         **kwargs):
    #     g_negloglik = self.g_negloglik(
    #         **kwargs
    #     )
    #     g_penalty = self.g_penalty(
    #         **kwargs
    #     )
    #     # this isn't quite right - only applies at 0
    #     penalty_dominated = np.abs(
    #         g_negloglik
    #     ) < (
    #         g_penalty
    #     )
    #     return (g_negloglik + g_penalty) * penalty_dominated

    def obj_mu(self, x):
        return self.objective(
            mu=x,
            kappa=self.params['kappa'],
            tau=self.params['tau'],
            omega=self.params['omega'],
            pi_kappa=self.params['pi_kappa'],
            pi_omega=self.params['pi_omega'])

    def obj_kappa(self, x):
        return self.objective(
            mu=self.params['mu'],
            kappa=x,
            tau=self.params['tau'],
            omega=self.params['omega'],
            pi_kappa=self.params['pi_kappa'],
            pi_omega=self.params['pi_omega'])

    def obj_tau(self, x):
        return self.objective(
            mu=self.params['mu'],
            kappa=self.params['kappa'],
            tau=x,
            omega=self.params['omega'],
            pi_kappa=self.params['pi_kappa'],
            pi_omega=self.params['pi_omega'])

    def obj_omega(self, x):
        return self.objective(
            mu=self.params['mu'],
            kappa=self.params['kappa'],
            tau=self.params['tau'],
            omega=x,
            pi_kappa=self.params['pi_kappa'],
            pi_omega=self.params['pi_omega'])

    def _setup_grad(self):
        self._grad_mu = autograd.grad(
            self.obj_mu, 0)
        self._grad_kappa = autograd.grad(
            self.obj_kappa, 0)
        self._grad_tau = autograd.grad(
            self.obj_tau, 0)
        self._grad_omega = autograd.grad(
            self.obj_omega, 0)

    def _fit(
            self,
            max_steps=3,
            step_iter=15,
            eps=1e-8,
            warm=False,
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
                fit['kappa'],
                method='TNC',
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

            new_fit['kappa'] = kappa

            if self._fit_tau:
                res = minimize(
                    self.obj_tau,
                    fit['tau'],
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

            if self._fit_omega:
                res = minimize(
                    self.obj_omega,
                    fit['omega'],
                    method='TNC',
                    jac=self._grad_omega,
                    bounds=self._omega_bounds,
                    callback=lambda x: self._debug_tee('omega_fit', x),
                    options=dict(
                        maxiter=step_iter,
                        disp=self.debug
                    )
                )
                omega = res.x
                new_fit['omega'] = omega

            # one-step mu update is possible for known noise structure
            # e.g. additive, but not in general
            res = minimize(
                self.obj_mu,
                fit['mu'],
                method='TNC',
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

            fit.update(**new_fit)

        self.params.update(fit)
        return fit

    # def _fit_path(
    #         self,
    #         param_vector=None,
    #         pi_kappa=0.0,
    #         pi_omega=1e-8,
    #         max_steps=50,
    #         step_iter=20,
    #         lr=0.1,
    #         gamma=0.9,
    #         eps=1e-8,
    #         backoff=0.75,
    #         warm_start=False,
    #         **kwargs
    #         ):
    #
    #     n_params = param_vector.size
    #     param_path = np.zeros((n_params, max_steps))
    #     pi_kappa_path = np.zeros(max_steps)
    #     pi_omega_path = np.zeros(max_steps)
    #     loglik_path = np.zeros(max_steps)
    #     dof_path = np.zeros(max_steps)
    #     aic_path = np.zeros(max_steps)
    #
    #     self._avg_sq_grad = avg_sq_grad = np.ones_like(param_vector)
    #
    #     for j in range(max_steps):
    #         loss = self._objective_packed(param_vector)
    #         best_loss = loss
    #         local_lr = lr
    #         best_param_vector = np.array(param_vector)
    #
    #         for i in range(step_iter):
    #             g_negloglik = self._grad_negloglik(param_vector)
    #             g_penalty = self._grad_penalty(param_vector, pi_kappa, pi_omega)
    #             g = g_negloglik + g_penalty
    #             self._debug_print(j, i, 'param', param_vector, 'grad', g)
    #             avg_sq_grad[:] = avg_sq_grad * gamma + g**2 * (1 - gamma)
    #
    #             velocity = lr * g * (
    #                     np.sqrt(avg_sq_grad) + eps
    #             ) / (0.1 * (i + j + 10.0))
    #             # watch out, nans
    #             velocity[np.logical_not(np.isfinite(velocity))] = 0.0
    #
    #             penalty_dominant = np.abs(
    #                 g_negloglik
    #             ) < (
    #                 self._penalty_weight_packed(pi_kappa, pi_omega)
    #             )
    #             velocity[penalty_dominant * (velocity == 0)] = 0.0
    #             new_param_vector = param_vector - velocity * local_lr
    #             # coefficients that pass through 0 must stop there
    #             new_param_vector[
    #                 np.abs(
    #                     np.sign(new_param_vector) -
    #                     np.sign(param_vector)
    #                 ) == 2
    #             ] = 0.0
    #             new_param_vector[:] = np.maximum(new_param_vector, self._param_floor)
    #             new_loss = self._objective_packed(new_param_vector)
    #             if new_loss < loss:
    #                 # print('good', loss, '=>', new_loss, local_lr)
    #                 loss = new_loss
    #                 param_vector = new_param_vector
    #                 self._param_vector = new_param_vector
    #             else:
    #                 # print('bad', loss, '=>', new_loss, local_lr)
    #                 local_lr = local_lr * backoff
    #                 new_param_vector = param_vector + backoff * (
    #                     new_param_vector - param_vector
    #                 )
    #                 loss = self._objective_packed(new_param_vector)
    #             if loss < best_loss:
    #                 best_param_vector = np.array(param_vector)
    #                 best_loss = loss
    #
    #             if local_lr < 1e-3:
    #                 self._debug_print('nope', j, i, max_steps)
    #                 break
    #
    #         this_loglik = -self._negloglik_packed(best_param_vector)
    #         this_dof = self._dof_packed(best_param_vector)
    #         param_path[:, j] = best_param_vector
    #         pi_kappa_path[j] = pi_kappa
    #         pi_omega_path[j] = pi_omega
    #         loglik_path[j] = this_loglik
    #         dof_path[j] = this_dof
    #         aic_path[j] = 2 * this_loglik - 2 * this_dof
    #
    #         # # regularisation parameter selection
    #         # all_grad = self._unpack(
    #         #     np.abs(
    #         #         grad_objective(best_param_vector) *
    #         #         (best_param_vector != 0.0)
    #         #     )
    #         # )
    #         # kappa_grad = all_grad['kappa']
    #         # log_omega_grad = all_grad.get('omega', None)
    #         #
    #         # if (
    #         #     np.random.random() < (
    #         #         np.sqrt(log_omega_grad.size) /
    #         #         (np.sqrt(kappa_grad.size) +
    #         #         np.sqrt(log_omega_grad.size))
    #         #         )):
    #         #     print('log_omega_grad', log_omega_grad)
    #         #     pi_omega += max(
    #         #         np.amin(log_omega_grad[log_omega_grad > 0])
    #         #         * j/max_steps,
    #         #         pi_omega * 0.1
    #         #     )
    #         # else:
    #         #     print('kappa_grad', kappa_grad)
    #         #     pi_kappa += max(
    #         #         np.amin(kappa_grad[kappa_grad > 0]) * j / max_steps,
    #         #         pi_kappa * 0.1
    #         #     )
    #     return FitHistory(
    #         param_path=param_path[:, :j],
    #         pi_kappa_path=pi_kappa_path[:j],
    #         pi_omega_path=pi_omega_path[:j],
    #         loglik_path=loglik_path[:j],
    #         dof_path=dof_path[:j],
    #         aic_path=aic_path[:j],
    #         unpack=lambda v: self._unpack(v),
    #         param_vector=best_param_vector,
    #         param=self._unpack(best_param_vector),
    #     )
    #
    # def coef_(self):
    #     return self._unpack(self._param_vector)
    #

class FitHistory(object):
    def __init__(self, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._keys = kwargs.keys()

    def __repr__(self):
        return 'FitHistory({param!r})'.format(param=self.param)
