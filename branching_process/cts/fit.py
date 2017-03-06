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


class ContinuousExact(object):
    def __init__(
            self,
            phis,
            fit_tau=False
            ):
        if hasattr(phis, 'integrate'):
            # assume a single kernel, instead of an array
            phis = [phis]
        self.phis = phis
        self.n_bases = len(phis)
        self.fit_tau = fit_tau

    def _pack(
            self,
            mu=1.0,
            kappas=0.0,
            taus=[]
            # log_omega=0.0,
            ):

        n_taus = self.n_bases * self.fit_tau
        packed = np.zeros(
            1 +
            self.n_bases +
            self.n_bases * self.fit_tau
            # n_steps
        )
        packed[0] = mu
        packed[1:self.n_bases + 1] = kappas
        packed[self.n_bases + 1:self.n_bases + n_taus + 1] = taus
        # packed[self.n_bases+n_taus + 1:] = log_omega
        return packed

    def _unpack(
            self,
            packed):
        """
        returns mu, kappas, taus, #log_omegas
        """
        n_taus = self.n_bases * self.fit_tau
        return (
            packed[0],
            packed[1:self.n_bases+1],
            packed[self.n_bases + 1:self.n_bases + n_taus + 1],
            # packed[self.n_bases + n_taus + 1:]
        )

    def _negloglik_packed(
            self,
            param_vector):
        return self.negloglik(*self._unpack(param_vector))

    def negloglik(
            self,
            mu,
            kappas,
            taus=0.0,
            log_omegas=0.0,
            ):
        endo_rate = np.dot(np.reshape(kappas, (1, -1)), X)
        lamb = endo_rate + mu * np.exp(log_omega)
        partial_loglik = loglik_poisson(lamb, y)
        return -np.sum(partial_loglik)

    def _penalty_weight_packed(
            self,
            pi_kappa=0.0,
            # pi_omega
            ):
        return self._pack(
            mu=0.0,
            kappas=pi_kappa,
            # log_omega=pi_omega
        ) / self.penalty_scale

    def _penalty_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            # pi_omega=0.0
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
            # pi_omega=0.0,
            ):
        """
        approximate self._dof_packed differentiably, using gradient information.
        """
        raise NotImplementedError()

    def _dof_packed(
            self,
            param_vector,
            pi_kappa=0.0,
            # pi_omega=0.0
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
            kappas=0.0,
            taus=0.0,
            # log_omegas=0.0,
            **kwargs
            ):
        return self._fit_packed(
            ts,
            self._pack(
                mu=mu,
                kappas=kappas,
                taus=taus,
                # log_omegas=log_omegas
            ),
            **kwargs
        )

    def _fit_packed(
            self,
            ts,
            param_vector=None,
            t0=0.0,
            t1=None,
            pi_kappa=0.0,
            # pi_omega=1e-8,
            # max_steps=None,
            # step_iter=50,
            # step_size=0.1,
            # gamma=0.9,
            eps=1e-8,
            # backoff=0.75
            ):
        if param_vector is None:
            param_vector = self._pack()

        self.t0 = t0
        self.t1 = t1 or np.maximum(ts)

        # Scale factors by the mean only, since they are assumed to be positive
        X_scale = X.mean(axis=1)
        y_scale = y.mean()

        # optimizer scale - not quite the same as penalty scale
        param_scale = self._pack(
            mu=y_scale * 0.5,
            kappas=X_scale,
            log_omega=1.0
        )
        # reweight penalty according to magnitude of predictors
        penalty_scale = self._pack(
            mu=1.0,  # unused
            kappas=X_scale,
            log_omega=1.0
        )
        param_floor = self._pack(
            mu=0.0,  # unused
            kappas=0.0,
            log_omega=-np.inf
        )

        n_params = param_vector.size
        param_path = np.zeros((n_params, max_steps))
        pi_kappa_path = np.zeros(max_steps)
        pi_omega_path = np.zeros(max_steps)
        loglik_path = np.zeros(max_steps)
        dof_path = np.zeros(max_steps)
        aic_path = np.zeros(max_steps)

        # Now, an idiotic gradient descent algorithm
        # Seeding by iteratively-reweighted least squares
        # or just least squares would be better
        grad_negloglik = grad(self._negloglik_packed, 0)
        grad_penalty = grad(self._penalty_packed, 0)
        grad_objective = grad(self._objective_packed, 0)

        avg_sq_grad = np.ones_like(param_vector)

        for j in range(max_steps):
            loss = self._objective_packed(param_vector)
            best_loss = loss
            local_step_size = step_size
            best_param_vector = np.array(param_vector)

            for i in range(step_iter):
                g_negloglik = grad_negloglik(param_vector)
                g_penalty = grad_penalty(param_vector, pi_kappa, pi_omega)
                g = g_negloglik + g_penalty
                avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)

                velocity = g/(np.sqrt(avg_sq_grad) + eps) / sqrt(i+1.0)
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
            # ideally should be randomly weight according
            # to sizes of those two damn vectors
            mu_grad, kappa_grad, log_omega_grad = self._unpack(
                np.abs(
                    grad_objective(best_param_vector) *
                    (best_param_vector != 0.0)
                )
            )
            if (
                np.random.random() < (
                    sqrt(log_omega_grad.size) /
                    (sqrt(kappa_grad.size) + sqrt(log_omega_grad.size))
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
        return self._unpack()
