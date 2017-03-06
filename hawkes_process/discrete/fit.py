"""
Poisson time series penalised likelihood regression.
"""

try:
    import autograd
    import autograd.numpy as np
    import autograd.scipy as sp
    from autograd import grad
    have_autograd = True
except ImportError as e:
    import numpy as np
    import scipy as sp
    have_autograd = False


from .design_discrete import history_expand
from .dist_discrete import loglik_poisson
from math import sqrt


def poisson_ts_model_fit_np(ts, basis):
    X = history_expand(ts, basis)
    y = np.asarray(ts[1:], dtype="float")
    n_bases = X.shape[0]
    n_steps = y.size

    # I could re-scale X and Y here
    # X_scaled = X / X_scale.reshape(-1,1)
    # y_scaled = y / y_scale

    def pack(mu=y.mean(), kappas=0.0, log_omega=0.0):
        packed = np.zeros(1 + n_bases + n_steps)
        packed[0] = mu
        packed[1:n_bases+1] = kappas
        packed[n_bases+1:] = log_omega
        return packed

    def unpack(packed):
        """
        returns mu, kappas, log_omegas
        """
        return packed[0], packed[1:n_bases+1], packed[n_bases+1:]

    def negloglik(param_vector):
        mu, kappas, log_omega = unpack(param_vector)
        endo_rate = np.dot(np.reshape(kappas, (1, -1)), X)
        lamb = endo_rate + mu * np.exp(log_omega)
        partial_loglik = loglik_poisson(lamb, y)
        return -np.sum(partial_loglik)

    def penalty_weight(pi_kappa, pi_omega):
        return pack(mu=0, kappas=pi_kappa, log_omega=pi_omega) / penalty_scale

    def penalty(param_vector, pi_kappa=0.0, pi_omega=0.0):
        # 2 different l_1 penalties
        return np.sum(
            np.abs(param_vector) *
            penalty_weight(pi_kappa, pi_omega)
        )

    def soft_dof(param_vector, pi_kappa=0.0, pi_omega=0.0):
        """
        approximate DOF differentiably, using gradient information.
        """
        raise NotImplementedError()

    def dof(param_vector, pi_kappa=0.0, pi_omega=0.0):
        """
        estimate DOF using a naive scale-unaware threshold
        """
        return np.sum(np.isclose(param_vector, 0))

    def objective(param_vector, pi_kappa=0.0, pi_omega=0.0):
        return negloglik(
            param_vector
        ) + penalty(
            param_vector, pi_kappa, pi_omega
        )

    def fit(
            param_vector=None,
            pi_kappa=0.0,
            pi_omega=1e-8,
            max_steps=y.size,
            step_iter=50,
            step_size=0.1,
            gamma=0.9,
            eps=1e-8,
            backoff=0.75):
        if param_vector is None:
            param_vector = pack()
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
        grad_negloglik = grad(negloglik, 0)
        grad_penalty = grad(penalty, 0)
        grad_objective = grad(objective, 0)

        avg_sq_grad = np.ones_like(param_vector)

        for j in range(max_steps):
            loss = objective(param_vector)
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
                    penalty_weight(pi_kappa, pi_omega)
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
                new_loss = objective(new_param_vector)
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
                    loss = objective(new_param_vector)
                if loss < best_loss:
                    best_param_vector = np.array(param_vector)
                    best_loss = loss

                if local_step_size < 1e-3:
                    print('nope', j, i, max_steps)
                    break

            this_loglik = -negloglik(best_param_vector)
            this_dof = dof(best_param_vector)
            param_path[:, j] = best_param_vector
            pi_kappa_path[j] = pi_kappa
            pi_omega_path[j] = pi_omega
            loglik_path[j] = this_loglik
            dof_path[j] = this_dof
            aic_path[j] = 2 * this_loglik - 2 * this_dof

            # regularisation parameter selection
            # ideally should be randomly weight according
            # to sizes of those two damn vectors
            mu_grad, kappa_grad, log_omega_grad = unpack(
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

        return dict(
            param_path=param_path,
            pi_kappa_path=pi_kappa_path,
            pi_omega_path=pi_omega_path,
            loglik_path=loglik_path,
            dof_path=dof_path,
            aic_path=aic_path
        )

    # Scale factors by the mean only, since they are assumed to be positive
    X_scale = X.mean(axis=1)
    y_scale = y.mean()

    # optimizer scale - not quite the same as penalty scale
    param_scale = pack(
        mu=y_scale * 0.5,
        kappas=X_scale,
        log_omega=1.0
    )
    # reweight penalty according to magnitude of predictors
    penalty_scale = pack(
        mu=1.0,  # unused
        kappas=X_scale,
        log_omega=1.0
    )
    param_floor = pack(
        mu=0.0,  # unused
        kappas=0.0,
        log_omega=-np.inf
    )

    return dict(
        pack=pack,
        unpack=unpack,
        negloglik=negloglik,
        penalty=penalty,
        objective=objective,
        dof=dof,
        fit=fit
    )
