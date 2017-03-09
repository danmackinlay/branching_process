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


def lam_hawkes(
        ts,
        mu,
        phi,
        eta=1.0,
        eval_ts=None,
        max_floats=1e8,
        phi_kwargs={},
        mu_kwargs={}):
    """
    True intensity of Hawkes process.
    Memory-hungry per default; could be improve with numba.
    """
    phi = influence.as_influence_kernel(phi)
    mu = influence.as_influence_kernel(mu)
    ts = np.asfarray(ts).ravel()
    if eval_ts is None:
        eval_ts = ts
    eval_ts = np.asfarray(eval_ts).ravel()
    if ((ts.size) * (eval_ts.size)) > max_floats:
        return _lam_hawkes_lite(
            ts=ts,
            mu=mu,
            phi=phi,
            eta=eta,
            eval_ts=eval_ts,
            phi_kwargs=phi_kwargs,
            mu_kwargs=mu_kwargs)
    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    endo = phi(deltas.ravel(), **phi_kwargs).reshape(deltas.shape) * mask
    return endo.sum(0) * eta + mu(deltas, **mu_kwargs)


def _lam_hawkes_lite(
        ts,
        eval_ts,
        mu,
        phi,
        eta=1.0,
        start=0.0,
        phi_kwargs={},
        mu_kwargs={}):
    """
    True intensity of hawkes process.
    Memory-lite version. CPU-hungry, could be improved with numba.

    Uses assignment so may need to be altered for differentiability.
    """
    endo = np.zeros_like(eval_ts)
    big_endo = np.zeros_like(eval_ts)
    deltas = np.zeros_like(ts)
    mask = np.zeros_like(ts)
    for i in range(eval_ts.size):
        deltas[:] = eval_ts[i] - ts
        mask[:] = deltas > 0.0
        endo[i] = np.sum(phi(deltas, **phi_kwargs) * mask)
        # possibly unneeded:
        big_endo[i] = np.sum(phi.integrate(deltas) * mask)
    return endo * eta + mu(eval_ts, **mu_kwargs)


def big_lam_hawkes(
        ts,
        eval_ts,
        mu,
        phi,
        eta=1.0,
        start=0.0,
        phi_kwargs={},
        mu_kwargs={}
        ):
    """
    True integrated intensity of hawkes process.
    since you are probably evaluating this only at one point,
    this is only available in vectorised high-memory version.
    """
    phi = influence.as_influence_kernel(phi)
    mu = influence.as_influence_kernel(mu)
    ts = np.asfarray(ts).ravel()

    deltas = eval_ts.reshape(1, -1) - ts.reshape(-1, 1)
    mask = deltas > 0.0
    big_endo = phi.integrate(
        deltas.ravel(),
        **phi_kwargs
    ).reshape(deltas.shape) * mask
    big_lam = (
        big_endo.sum(0) * eta +
        mu.integrate(eval_ts, **mu_kwargs) -
        mu.integrate(start, **mu_kwargs)
    )
    return big_lam


class ContinuousExact(object):
    def __init__(
            self,
            phi=None,
            n_phi_bases=1,
            n_omega_bases=0,
            debug=False
            ):
        if phi is None:
            phi = influence.MaxwellKernel(n_bases=5)
        else:
            phi = influence.as_influence_kernel(phi, n_bases=n_phi_bases)
        self.phi = phi
        self.n_phi_bases = phi.n_bases
        self.n_omega_bases = n_omega_bases
        self.debug = debug

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _pack(
            self,
            mu=1.0,
            kappa=0.1,
            tau=1.0,
            log_omega=0.0,
            **kwargs
            ):
        n_tau = self.n_phi_bases * self._fit_tau
        n_omega = self.n_omega_bases * self._fit_omega
        packed = np.zeros(
            1 +
            self.n_phi_bases +
            n_tau +
            n_omega
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
        n_omega = self.n_omega_bases * self._fit_omega
        unpacked = {
            'mu': packed[0],
            'kappa': packed[1:self.n_phi_bases+1],
        }
        if n_tau > 0:
            unpacked['tau'] = packed[
                self.n_phi_bases + 1:self.n_phi_bases + n_tau + 1
            ]
        if n_omega > 0:
            unpacked['omega'] = packed[
                self.n_phi_bases + n_tau + 1:
            ]
        return unpacked

    def _negloglik_packed(
            self,
            param_vector, **kwargs):
        return self.negloglik(
            self._ts,
            self._evalts,
            self.phi,
            **self._unpack(param_vector),
            **kwargs)

    def negloglik(
            self,
            ts,
            eval_ts=None,
            phi=None,
            mu=1.0,
            eta=1.0,
            start=None,
            end=None,
            log_omega=[],
            **phi_kwargs):
        if phi is None:
            phi = self.phi
        if end is None:
            end = getattr(self, '_t_end', ts[-1])
        if start is None:
            start = getattr(self, '_t_start', 0.0)
        lam = lam_hawkes(
            ts=ts,
            mu=mu,
            phi=phi,
            eta=eta,
            eval_ts=eval_ts,
            phi_kwargs=phi_kwargs,
            mu_kwargs={}
        )
        big_lam = big_lam_hawkes(
            ts=ts,
            mu=mu,
            phi=phi,
            start=start,
            eta=eta,
            eval_ts=np.array(end),
            phi_kwargs=phi_kwargs,
            mu_kwargs={}
        )
        negloglik = big_lam - np.sum(np.log(lam))
        return negloglik

    def loglik(
            self,
            *args,
            **kwargs):
        return -self.loglik(*args, **kwargs)

    def _penalty_weight_packed(
            self,
            pi_kappa=0.0,
            pi_omega=0.0
            ):
        return self._pack(
            mu=0.0,
            kappa=pi_kappa,
            log_omega=pi_omega
        )  # / self.penalty_scale

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
                pi_omega
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
            fit_tau=False,
            fit_omega=False,
            **kwargs
            ):
        self._fit_tau = fit_tau
        self._fit_omega = fit_omega
        return self._fit_packed(
            ts,
            self._pack(
                **kwargs
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
            _evalts = np.append(
                ts[ts > self._t_start],
                [self._t_end]
            )
        else:
            _evalts = ts[
                np.logical_and(
                    ts >= self._t_start,
                    ts < self._t_end
                )
            ]

        self._evalts = _evalts

        param_floor = self._pack(
            mu=0.0,  # unused?
            kappa=0.0,
            log_omega=-np.inf
        )

        n_params = param_vector.size
        param_path = np.zeros((n_params, max_steps))
        pi_kappa_path = np.zeros(max_steps)
        pi_omega_path = np.zeros(max_steps)
        loglik_path = np.zeros(max_steps)
        dof_path = np.zeros(max_steps)
        aic_path = np.zeros(max_steps)

        grad_negloglik = autograd.grad(self._negloglik_packed, 0)
        grad_penalty = autograd.grad(self._penalty_packed, 0)
        # grad_objective = autograd.grad(self._objective_packed, 0)

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
                self._debug_print(param_vector, g)
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
                    self._debug_print('nope', j, i, max_steps)
                    break

            this_loglik = -self._negloglik_packed(best_param_vector)
            this_dof = self._dof_packed(best_param_vector)
            param_path[:, j] = best_param_vector
            pi_kappa_path[j] = pi_kappa
            pi_omega_path[j] = pi_omega
            loglik_path[j] = this_loglik
            dof_path[j] = this_dof
            aic_path[j] = 2 * this_loglik - 2 * this_dof

            # # regularisation parameter selection
            # all_grad = self._unpack(
            #     np.abs(
            #         grad_objective(best_param_vector) *
            #         (best_param_vector != 0.0)
            #     )
            # )
            # kappa_grad = all_grad['kappa']
            # log_omega_grad = all_grad.get('omega', None)
            #
            # if (
            #     np.random.random() < (
            #         np.sqrt(log_omega_grad.size) /
            #         (np.sqrt(kappa_grad.size) +
            #         np.sqrt(log_omega_grad.size))
            #         )):
            #     print('log_omega_grad', log_omega_grad)
            #     pi_omega += max(
            #         np.amin(log_omega_grad[log_omega_grad > 0])
            #         * j/max_steps,
            #         pi_omega * 0.1
            #     )
            # else:
            #     print('kappa_grad', kappa_grad)
            #     pi_kappa += max(
            #         np.amin(kappa_grad[kappa_grad > 0]) * j / max_steps,
            #         pi_kappa * 0.1
            #     )
        del(self._ts)
        del(self._evalts)
        return Fit(
            param_path=param_path[:, :j],
            pi_kappa_path=pi_kappa_path[:j],
            pi_omega_path=pi_omega_path[:j],
            loglik_path=loglik_path[:j],
            dof_path=dof_path[:j],
            aic_path=aic_path[:j],
            unpack=lambda v: self._unpack(v),
            param_vector=param_vector,
            param=self._unpack(best_param_vector),
        )

    def coef_(self):
        return self._unpack(self._param_vector)


class Fit(object):
    def __init__(self, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._keys = kwargs.keys()
