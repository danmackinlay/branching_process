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

def intensity_hawkes(
        timestamps,
        mu,
        phi,
        eta=1.0,
        eval_timestamps=None,
        sort=True,
        max_floats=1e8,
        **kwargs):
    """
    True intensity of Hawkes process.
    Memory-hungry per default; could be improved, with numba.
    """
    timestamps = np.asarray(timestamps).ravel()
    if sort:
        timestamps = np.sort(timestamps)
    if eval_timestamps is None:
        eval_timestamps = timestamps
        if sort:
            eval_timestamps = np.sort(eval_timestamps)
    eval_timestamps = np.asarray(eval_timestamps).ravel()
    if ((timestamps.size) * (eval_timestamps.size)) > max_floats:
        return _intensity_hawkes_lite(
            timestamps=timestamps,
            mu=mu,
            phi=phi,
            eta=eta,
            eval_timestamps=eval_timestamps,
            **kwargs)
    deltas = eval_timestamps.reshape(1, -1) - timestamps.reshape(-1, 1)
    mask = deltas > 0.0
    endo = phi(deltas.ravel()).reshape(deltas.shape) * mask
    lambdas = endo.sum(0) * eta + mu
    return lambdas


def _intensity_hawkes_lite(
        timestamps,
        eval_timestamps,
        mu,
        phi,
        eta=1.0,
        **kwargs):
    """
    True intensity of hawkes process.
    Memory-lite version. CPU-hungry, could be improved with numba.
    """
    endo = np.zeros_like(eval_timestamps)
    deltas = np.zeros_like(timestamps)
    mask = np.zeros_like(timestamps)
    for i in range(eval_timestamps.size):
        deltas[:] = eval_timestamps[i] - timestamps
        mask[:] = deltas > 0.0
        endo[i] = np.sum(phi(deltas) * mask)
    return endo * eta + mu


class ContinuousExact(object):
    def __init__(
            self,
            phi=None,
            n_phi_bases=1,
            n_omega_bases=0,
            ):
        if phi is None:
            phi = influence.MaxwellKernel(n_bases=5)
        else:
            phi = influence.as_influence_kernel(phi, n_bases=n_phi_bases)
        self.phi = phi
        self.n_phi_bases = phi.n_bases
        self.n_omega_bases = n_omega_bases

    def _pack(
            self,
            mu=1.0,
            kappa=0.0,
            tau=1.0,
            log_omega=0.0,
            ):

        n_tau = self.n_phi_bases * self._fit_tau
        n_omegas = self.n_omega_bases * self._fit_omega
        packed = np.zeros(
            1 +
            self.n_phi_bases +
            n_tau +
            n_omegas
        )
        packed[0] = mu
        packed[1:self.n_phi_bases + 1] = kappa
        packed[self.n_phi_bases + 1:self.n_phi_bases + n_tau + 1] = tau
        packed[self.n_phi_bases + n_tau + 1:] = log_omega
        return packed

    def _unpack(
            self,
            packed):
        """
        returns mu, kappa, tau, #log_omega
        """
        n_tau = self.n_phi_bases * self._fit_tau
        return (
            packed[0],
            packed[1:self.n_phi_bases+1],
            packed[self.n_phi_bases + 1:self.n_phi_bases + n_tau + 1],
            packed[self.n_phi_bases + n_tau + 1:]
        )

    def _negloglik_packed(
            self,
            param_vector):
        return self.negloglik(
            self._ts,
            self._evalpts,
            self.phi,
            *self._unpack(param_vector))

    def negloglik(
            self,
            ts,
            evalpts=None,
            phi=None,
            mu=1.0,
            kappa=None,
            tau=0.0,
            log_omega=0.0,
            ):
        if phi is None:
            phi = self.phi
        if kappa is None:
            kappa = np.ones(phi.n_omega_bases) / phi.n_omega_bases
        lam = phi(ts, tau=tau) + mu
        big_lam = phi.integral(self._t_end) - phi.integral(self._t_start)
        negloglik = big_lam - np.sum(np.log(lam))
        return negloglik

    def _penalty_weight_packed(
            self,
            pi_kappa=0.0,
            pi_omega=0.0
            ):
        return self._pack(
            mu=0.0,
            kappa=pi_kappa,
            # log_omega=pi_omega
        ) / self.penalty_scale

    def _penalty_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            pi_omega=0.0
            ):
        # 2 different l_1 penalties
        return np.sum(
            np.abs(param_vector) *
            self._penalty_weight_packed(
                pi_kappa,
                # pi_omega
            )
        )

    def _soft_dof_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            pi_omega=0.0,
            ):
        """
        approximate self._dof_packed differentiably,
        using gradient information.
        """
        raise NotImplementedError()

    def _dof_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            pi_omega=0.0
            ):
        """
        estimate self._dof_packed using a naive scale-unaware threshold
        """
        return np.sum(np.isclose(param_vector, 0))

    def _objective_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            pi_omega=0.0):
        return self._negloglik_packed(
            param_vector
        ) + self._penalty_packed(
            param_vector,
            pi_kappa,
            # pi_omega
        )

    def fit(
            self,
            ts,
            mu=0.0,
            kappa=0.0,
            tau=0.0,
            log_omega=0.0,
            fit_tau=False,
            fit_omega=False,
            **kwargs
            ):
        self._fit_tau = fit_tau
        self._fit_omega = fit_omega
        return self._fit_packed(
            ts,
            self._pack(
                mu=mu,
                kappa=kappa,
                tau=tau,
                log_omega=log_omega
            ),
            **kwargs
        )

    def _fit_packed(
            self,
            ts,
            param_vector=None,
            t_start=0.0,
            t_end=None,
            pi_kappa=0.0,
            pi_omega=1e-8,
            max_steps=50,
            step_iter=50,
            step_size=0.1,
            gamma=0.9,
            eps=1e-8,
            backoff=0.75,
            warm_start=False,
            ):

        if warm_start:
            param_vector = self._param_vector
        else:
            if param_vector is None:
                param_vector = self._pack()
        self._param_vector = param_vector

        self._t_start = t_start
        self._t_end = t_end or ts[-1]
        self._ts = ts

        # Full likelihood is best evaluated at the end also
        if ts[-1] < self._t_end:
            _evalpts = np.append(
                ts[ts > self._t_start],
                [self._t_end]
            )
        else:
            _evalpts = ts[
                np.logical_and(
                    ts >= self._t_start,
                    ts < self._t_end
                )
            ]

        self._evalpts = _evalpts

        param_floor = self._pack(
            mu=0.0,  # unused
            kappa=0.0,
            log_omega=-np.inf
        )

        n_params = param_vector.size
        self._param_path = param_path = np.zeros((n_params, max_steps))
        self._pi_kappa_path = pi_kappa_path = np.zeros(max_steps)
        self._pi_omega_path = pi_omega_path = np.zeros(max_steps)
        self._loglik_path = loglik_path = np.zeros(max_steps)
        self._dof_path = dof_path = np.zeros(max_steps)
        self._aic_path = aic_path = np.zeros(max_steps)

        grad_negloglik = autograd.grad(self._negloglik_packed, 0)
        grad_penalty = autograd.grad(self._penalty_packed, 0)
        grad_objective = autograd.grad(self._objective_packed, 0)

        self._avg_sq_grad = avg_sq_grad = np.ones_like(param_vector)

        for j in range(max_steps):
            loss = self._objective_packed(param_vector)
            best_loss = loss
            local_step_size = step_size
            best_param_vector = np.array(param_vector)

            for i in range(step_iter):
                g_negloglik = grad_negloglik(param_vector)
                g_penalty = grad_penalty(param_vector, pi_kappa, pi_omega)
                g = g_negloglik + g_penalty
                avg_sq_grad[:] = avg_sq_grad * gamma + g**2 * (1 - gamma)

                velocity = g/(np.sqrt(avg_sq_grad) + eps) / np.sqrt(i+1.0)
                # watch out, nans
                velocity[np.logical_not(np.isfinite(velocity))] = 0.0

                penalty_dominant = np.abs(
                    g_negloglik
                ) < (
                    self._penalty_weight_packed(pi_kappa, pi_omega)
                )
                velocity[penalty_dominant * (velocity == 0)] = 0.0
                new_param_vector = param_vector - velocity * local_step_size
                # coefficients that pass through 0 must stop there
                new_param_vector[
                    np.abs(
                        np.sign(new_param_vector) -
                        np.sign(param_vector)
                    ) == 2
                ] = 0.0
                new_param_vector[:] = np.maximum(new_param_vector, param_floor)
                new_loss = self._objective_packed(new_param_vector)
                if new_loss < loss:
                    # print('good', loss, '=>', new_loss, local_step_size)
                    loss = new_loss
                    param_vector = new_param_vector
                    self._param_vector = new_param_vector
                else:
                    # print('bad', loss, '=>', new_loss, local_step_size)
                    local_step_size = local_step_size * backoff
                    new_param_vector = param_vector + backoff * (
                        new_param_vector - param_vector
                    )
                    loss = self._objective_packed(new_param_vector)
                if loss < best_loss:
                    best_param_vector = np.array(param_vector)
                    best_loss = loss

                if local_step_size < 1e-3:
                    print('nope', j, i, max_steps)
                    break

            this_loglik = -self._negloglik_packed(best_param_vector)
            this_dof = self._dof_packed(best_param_vector)
            param_path[:, j] = best_param_vector
            pi_kappa_path[j] = pi_kappa
            pi_omega_path[j] = pi_omega
            loglik_path[j] = this_loglik
            dof_path[j] = this_dof
            aic_path[j] = 2 * this_loglik - 2 * this_dof

            # regularisation parameter selection
            # ideally should be weighted according
            # to sizes of those two damn vectors
            mu_grad, kappa_grad, log_omega_grad = self._unpack(
                np.abs(
                    grad_objective(best_param_vector) *
                    (best_param_vector != 0.0)
                )
            )
            if (
                np.random.random() < (
                    np.sqrt(log_omega_grad.size) /
                    (np.sqrt(kappa_grad.size) + np.sqrt(log_omega_grad.size))
                    )):
                print('log_omega_grad', log_omega_grad)
                pi_omega += max(
                    np.amin(log_omega_grad[log_omega_grad > 0])
                    * j/max_steps,
                    pi_omega * 0.1
                )
            else:
                print('kappa_grad', kappa_grad)
                pi_kappa += max(
                    np.amin(kappa_grad[kappa_grad > 0]) * j / max_steps,
                    pi_kappa * 0.1
                )
        return self

    def coef_(self):
        return self._unpack(self._param_vector)
