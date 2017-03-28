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
    import autograd.scipy as sp
    have_autograd = True
except ImportError as e:
    import numpy as np
    import scipy as sp
    have_autograd = False

from . import influence
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

    def negloglik(
            self,
            mu=1.0,
            kappa=0.0,
            tau=None,
            omega=None,
            **kwargs):

        return -model.loglik(
            ts=self._ts,
            eval_ts=self._eval_ts,
            mu=mu,
            phi_kernel=self.phi_kernel,
            mu_kernel=self.mu_kernel,
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

    def objective(
            self,
            mu=1.0,
            kappa=0.0,
            tau=None,
            omega=None,
            pi_kappa=0.0,
            pi_omega=1e-8,
            **kwargs):

        g_negloglik = self.negloglik(
            mu=mu,
            kappa=kappa,
            tau=tau,
            omega=omega,
            pi_kappa=pi_kappa,
            pi_omega=pi_omega,
            **kwargs
        )
        g_penalty = self.penalty(
            mu=mu,
            kappa=kappa,
            tau=tau,
            omega=omega,
            pi_kappa=pi_kappa,
            pi_omega=pi_omega,
            **kwargs
        )
        penalty_dominated = np.abs(
            g_negloglik
        ) < (
            g_penalty
        )
        return (g_negloglik + g_penalty) * penalty_dominated

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
            param_vector=None,):

        self._t_start = t_start
        self._t_end = t_end or ts[-1]
        self._ts = ts
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
                mu_kernel = influence.LinearStepKernel(
                    t_start=self._t_start,
                    t_end=self._t_end,
                    n_bases=n_mu_bases)
            else:
                mu_kernel = influence.ConstKernel()
        self.mu_kernel = mu_kernel
        self.n_mu_bases = n_mu_bases

        self._fit_tau = fit_tau
        self._fit_omega = fit_omega
        self.n_tau = self.n_phi_bases * self._fit_tau
        self.n_omega = self.n_mu_bases * self._fit_omega

        if param_vector is None:
            param_vector = self._pack()
        self._param_vector = param_vector

        self._param_floor = self._pack(
            mu=0.0,  # unused?
            mu_kwargs=dict(
                kappa=0.0
            ),
            phi_kwargs=dict(
                kappa=0.0
            )
        )
        self._grad_mu = autograd.grad(
            self._negloglik_packed, 0)
        self._grad_kappa = autograd.grad(
            self._negloglik_packed, 1)
        self._grad_tau = autograd.grad(
            self._negloglik_packed, 2)
        self._grad_omega = autograd.grad(
            self._negloglik_packed, 3)
        # self._hessian_diag_negloglik = autograd.elementwise_grad(
        #     self._grad_negloglik, 0)

    def _teardown_graphs(
            self):

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

        param_kwargs = dict()
        for k in [
            'mu', 'tau', 'kappa', 'omega', 'pi_kappa', 'pi_omega'
        ]:
            v = kwargs.get(k, None)
            if v is not None:
                param_kwargs[k] = v
        return self._fit(
            **param_kwargs,
            **kwargs
        )

    def _fit(
            self,
            mu=1.0,
            kappa=0.0,
            tau=None,
            omega=None,
            pi_kappa=0.0,
            pi_omega=1e-8,
            max_steps=10,
            eps=1e-8,
            **kwargs
            ):
        """
        fit by Truncated Newton in each coordinate
        """

        for i in range(max_steps):
            self._debug_print(i, 'param', mu, tau, kappa, omega, 'grad', g)
            res = scipy.minimize(


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
